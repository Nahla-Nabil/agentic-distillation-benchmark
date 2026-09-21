"""Batched generation for the harness, without changing the harness.

harness.executor.run_task() is written as one task at a time: it calls
`model_fn(messages) -> text`, waits, then decides the next step. Evaluating
221 tasks that way keeps a GPU almost idle (one short generation at a time)
and is what made every condition cost ~40 minutes.

BatchedModelFn keeps that interface. Many tasks run in worker threads; each
calls the same `model_fn(messages)`, which parks the request; one server
thread gathers the pending requests into a single left-padded
`generate_batch(list_of_messages)` call and hands every caller its own text
back. run_task() and every metric are untouched.

Dispatch rule: send the batch as soon as every ACTIVE worker has a request
waiting (or `max_batch` are waiting), or after `max_wait_s` — so a small run
never waits for a full batch, and workers that finish early do not stall the
rest.

With greedy decoding the text is the same as the one-at-a-time path up to
floating-point differences caused by padding; those can flip a near-tie in
rare cases, so compare a batched run with a sequential one before mixing the
two in one table (see scripts/check_batched_eval.py).
"""

import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

_STOP = object()


class BatchedModelFn:
    def __init__(
        self,
        generate_batch: Callable[[list[list[dict[str, Any]]]], list[str]],
        max_batch: int = 16,
        max_wait_s: float = 0.25,
    ):
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        self.max_batch = max_batch
        self._generate_batch = generate_batch
        self._max_wait_s = max_wait_s
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._active = 0
        self.batch_sizes: list[int] = []  # for diagnostics and tests
        self._thread = threading.Thread(target=self._serve, name="batched-model-fn", daemon=True)
        self._thread.start()

    # ---- the ModelFn interface run_task() expects --------------------------
    def __call__(self, messages: list[dict[str, Any]]) -> str:
        future: Future = Future()
        self._queue.put((messages, future))
        return future.result()

    # ---- running many tasks through it -------------------------------------
    def run_concurrently(self, fn: Callable[[Any], Any], items: list[Any]) -> list[Any]:
        """fn(item) for every item on up to `max_batch` threads; results keep the input order."""

        def job(item):
            with self._lock:
                self._active += 1
            try:
                return fn(item)
            finally:
                with self._lock:
                    self._active -= 1

        with ThreadPoolExecutor(max_workers=self.max_batch) as pool:
            return list(pool.map(job, items))

    def close(self) -> None:
        self._queue.put(_STOP)
        self._thread.join(timeout=10)

    # ---- server thread -------------------------------------------------------
    def _serve(self) -> None:
        pending: list[tuple[list[dict[str, Any]], Future]] = []
        oldest = 0.0
        while True:
            try:
                item = self._queue.get(timeout=0.02)
            except queue.Empty:
                item = None
            if item is _STOP:
                for _, future in pending:
                    future.set_exception(RuntimeError("BatchedModelFn was closed with requests waiting"))
                return
            if item is not None:
                if not pending:
                    oldest = time.monotonic()
                pending.append(item)
            if not pending:
                continue
            with self._lock:
                active = self._active
            ready = len(pending) >= min(self.max_batch, max(active, 1))
            timed_out = time.monotonic() - oldest >= self._max_wait_s
            if ready or timed_out:
                batch, pending = pending[: self.max_batch], pending[self.max_batch :]
                if pending:
                    oldest = time.monotonic()
                self._dispatch(batch)

    def _dispatch(self, batch: list[tuple[list[dict[str, Any]], Future]]) -> None:
        self.batch_sizes.append(len(batch))
        try:
            texts = self._generate_batch([messages for messages, _ in batch])
            if len(texts) != len(batch):
                raise RuntimeError(f"generate_batch returned {len(texts)} texts for {len(batch)} requests")
        except Exception as e:  # noqa: BLE001 — every waiting caller must be released with the error
            for _, future in batch:
                future.set_exception(e)
            return
        for (_, future), text in zip(batch, texts, strict=True):
            future.set_result(text)


def configure_greedy(model) -> None:
    """Greedy decoding, set once on the generation config (not per call, so
    transformers does not warn on every step). Also unsets the checkpoint's
    default max_length, which would otherwise clash with max_new_tokens."""
    model.generation_config.max_length = None
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None


def make_batch_generate_fn(model, tokenizer, max_new_tokens: int = 256):
    """(model, tokenizer) -> generate_batch(list of chat-message lists) -> list of
    completions. Prompts are left-padded; each completion is decoded up to and
    including its first end-of-turn token, exactly like the single-prompt path,
    so padding after a short answer never leaks into the text.

    The returned function records, per call, `stats` (batch size, seconds, tokens
    generated per row, how many rows used the whole max_new_tokens budget) and keeps
    the first few texts of rows that hit the budget in `capped_samples`. A batch only
    ends when its slowest row does, so a single row that never emits its end-of-turn
    token makes the whole batch run to max_new_tokens; these numbers show whether that
    is what limits speed."""
    configure_greedy(model)
    stats: list[dict[str, Any]] = []
    capped_samples: list[str] = []

    def generate_batch(batch_messages: list[list[dict[str, Any]]]) -> list[str]:
        import torch

        started = time.monotonic()

        prompts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in batch_messages
        ]
        previous_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        try:
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        finally:
            tokenizer.padding_side = previous_side
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
        prompt_len = encoded["input_ids"].shape[1]

        eos = getattr(model.generation_config, "eos_token_id", None)
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) if eos is not None else set()
        if tokenizer.eos_token_id is not None:
            eos_ids.add(tokenizer.eos_token_id)

        completions = []
        lengths = []
        for row in output_ids:
            new_tokens = row[prompt_len:].tolist()
            cut = len(new_tokens)
            for i, token in enumerate(new_tokens):
                if token in eos_ids:
                    cut = i + 1
                    break
            lengths.append(cut)
            text = tokenizer.decode(new_tokens[:cut])
            if cut >= max_new_tokens and len(capped_samples) < 5:
                capped_samples.append(text[-600:])
            completions.append(text)
        stats.append({
            "batch": len(batch_messages),
            "seconds": round(time.monotonic() - started, 1),
            "prompt_tokens": int(prompt_len),
            "new_tokens": lengths,
            "hit_budget": sum(n >= max_new_tokens for n in lengths),
        })
        return completions

    generate_batch.stats = stats
    generate_batch.capped_samples = capped_samples
    return generate_batch
