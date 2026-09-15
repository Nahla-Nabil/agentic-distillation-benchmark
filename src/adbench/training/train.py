"""The single Colab-run training script for all three conditions.

    python -m adbench.training.train --condition base
    python -m adbench.training.train --condition sft_only
    python -m adbench.training.train --condition distilled
    python -m adbench.training.train --condition distilled --sweep higher_kd_temp

All three conditions share this one script (configs/experiment.yaml's
`conditions` list no longer points at per-condition trainer modules) so
"sft_only" and "distilled" are guaranteed to differ *only* in the loss
function — same data, same LoRA config, same optimizer/schedule — per the
README's three-way-comparison design. "base" does no training at all: it
loads the student, attaches a freshly-initialized (mathematically
untrained, near-identity) LoRA adapter, and saves immediately — see
save_checkpoint()'s docstring for why, rather than special-casing "no
checkpoint" for base elsewhere in the pipeline.

STRUCTURE — heavy deps (unsloth, bitsandbytes, and torch's actual model
forward passes) are only touched inside functions that need them, imported
lazily rather than at module load time. Everything above that layer —
config parsing/resolution, the per-step loss-computation wiring
(training_step), and training-example formatting (format_training_example)
— is plain Python + CPU torch, so it's unit-tested locally (tests/test_train.py)
using fake stand-in models and a stub tokenizer, with NO Unsloth, NO GPU, and
no real Qwen weights. Only the actual multi-epoch training loop over real
14B/4B models (run_training_loop, train_condition, main) needs Colab — that
code is complete and meant to run as-is there, but by construction can't be
exercised by a local test suite; treat it as reviewed-by-reading, not
reviewed-by-running, until it's run once on Colab.
"""

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from adbench.training.losses import KDLossConfig, combined_loss

REPO_ROOT = Path(__file__).resolve().parents[3]
CONDITIONS = ("base", "sft_only", "distilled")


# --------------------------------------------------------------------------
# Config parsing / resolution — pure Python, no torch. Fully unit-tested.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    """Everything one condition's run needs, resolved from
    configs/experiment.yaml's `training:` block (with a sweep's overrides
    applied, if any) plus the condition-specific KD weighting rule."""

    condition: str
    checkpoint_dir: Path
    seed: int
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    logging_steps: int
    save_steps: int
    kd: KDLossConfig


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_by_dotted_path(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Mutates `d` in place: _set_by_dotted_path(d, "kd.temperature", 4.0)
    sets d["kd"]["temperature"] = 4.0. Used to apply one sweep entry's
    `overrides` (configs/experiment.yaml: training.sweep) on top of the base
    training config."""
    keys = dotted_key.split(".")
    node = d
    for k in keys[:-1]:
        if k not in node:
            raise KeyError(f"Sweep override path {dotted_key!r}: no key {k!r}.")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"Sweep override path {dotted_key!r}: no key {keys[-1]!r}.")
    node[keys[-1]] = value


def resolve_checkpoint_dir(experiment_config: dict[str, Any], condition: str) -> Path:
    for c in experiment_config["conditions"]:
        if c["id"] == condition:
            return REPO_ROOT / c["checkpoint"]
    raise ValueError(
        f"No entry for condition {condition!r} in configs/experiment.yaml's "
        f"conditions list; known ids: {[c['id'] for c in experiment_config['conditions']]}."
    )


def resolve_training_config(
    experiment_config: dict[str, Any],
    condition: str,
    sweep_name: str | None = None,
) -> TrainingConfig:
    """Build the resolved TrainingConfig for one condition's run.

    - condition="base": KD is irrelevant (no training happens at all); the
      returned config's `kd` field is a harmless placeholder
      (kd_weight=0.0, sft_weight=1.0) that train_condition() never uses for
      "base" — it exists so this function always returns a well-typed
      TrainingConfig regardless of condition.
    - condition="sft_only": `kd` is FORCED to KDLossConfig(kd_weight=0.0,
      sft_weight=1.0) regardless of what configs/experiment.yaml's
      training.kd block or a --sweep says. This is deliberate, not a
      shortcut: sft_only must never depend on the teacher by construction,
      not by trusting the config file to have kd_weight=0 — see the
      module docstring on why sft_only/distilled must differ *only* in
      this weighting.
    - condition="distilled": `kd` comes from training.kd (temperature,
      kd_weight, sft_weight), with `sweep_name`'s overrides applied on top
      if given.

    All the non-KD hyperparameters (learning rate, epochs, batch size, ...)
    are shared across sft_only and distilled — and a sweep can override
    those too (e.g. the "lower_lr" sweep entry), since the point of the
    sweep mechanism is exploring the distilled condition's hyperparameters,
    not just its KD weighting.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}.")

    training_block = copy.deepcopy(experiment_config["training"])
    sweep_list = training_block.pop("sweep", [])

    if sweep_name is not None:
        matches = [s for s in sweep_list if s["name"] == sweep_name]
        if not matches:
            available = [s["name"] for s in sweep_list]
            raise ValueError(f"Unknown sweep {sweep_name!r}; available: {available}.")
        for dotted_key, value in matches[0].get("overrides", {}).items():
            _set_by_dotted_path(training_block, dotted_key, value)

    kd_block = training_block.pop("kd")
    if condition == "sft_only":
        kd = KDLossConfig(kd_weight=0.0, sft_weight=1.0)
    elif condition == "distilled":
        kd = KDLossConfig(
            temperature=kd_block["temperature"],
            kd_weight=kd_block["kd_weight"],
            sft_weight=kd_block["sft_weight"],
        )
    else:  # base
        kd = KDLossConfig(kd_weight=0.0, sft_weight=1.0)

    return TrainingConfig(
        condition=condition,
        checkpoint_dir=resolve_checkpoint_dir(experiment_config, condition),
        seed=training_block["seed"],
        learning_rate=training_block["learning_rate"],
        num_train_epochs=training_block["num_train_epochs"],
        per_device_train_batch_size=training_block["per_device_train_batch_size"],
        gradient_accumulation_steps=training_block["gradient_accumulation_steps"],
        warmup_ratio=training_block["warmup_ratio"],
        weight_decay=training_block["weight_decay"],
        lr_scheduler_type=training_block["lr_scheduler_type"],
        logging_steps=training_block["logging_steps"],
        save_steps=training_block["save_steps"],
        kd=kd,
    )


# --------------------------------------------------------------------------
# Per-step loss wiring — model-agnostic (works with a real Unsloth/PEFT
# model, or a tiny fake nn.Module in tests). Fully unit-tested with dummy
# tensors and fake models; no Unsloth/GPU needed for that.
# --------------------------------------------------------------------------

def training_step(student, teacher, input_ids, attention_mask, labels, cfg: KDLossConfig):
    """One training step's loss computation. `student` is called with
    gradients enabled; `teacher` (if cfg.kd_weight > 0) is called under
    torch.no_grad() so it's never updated — a real distillation setup must
    never backprop into the teacher.

    student/teacher: callables (nn.Module or a PreTrainedModel) invoked as
        model(input_ids=input_ids, attention_mask=attention_mask), returning
        either an object with a `.logits` attribute (HF's CausalLMOutput
        convention) or a raw logits tensor directly — both are accepted so
        this works with the real Unsloth-wrapped model AND a plain
        nn.Module test double.

    Returns (loss, components) exactly as training/losses.py::combined_loss
    does — the caller (run_training_loop) calls loss.backward() and logs
    components.
    """
    import torch  # deferred: only needed once an actual step runs

    def _logits_of(output):
        return output.logits if hasattr(output, "logits") else output

    student_logits = _logits_of(student(input_ids=input_ids, attention_mask=attention_mask))

    teacher_logits = None
    if cfg.kd_weight > 0:
        if teacher is None:
            raise ValueError("cfg.kd_weight > 0 but no teacher model was provided.")
        with torch.no_grad():
            teacher_logits = _logits_of(teacher(input_ids=input_ids, attention_mask=attention_mask))

    return combined_loss(student_logits, teacher_logits, labels, cfg)


# --------------------------------------------------------------------------
# Training-example formatting — glaive record -> tokenized {input_ids,
# attention_mask, labels}, in Qwen's <tool_call> convention (matching
# harness/executor.py's parser, so training and eval speak the same
# protocol). Testable with a stub tokenizer (no network) or the real Qwen
# tokenizer (network, no GPU) — see tests/test_train.py for both.
# --------------------------------------------------------------------------

def format_training_example(record: dict[str, Any], tokenizer, tool_registry) -> dict[str, list[int]]:
    """One data/prepare.py-produced record -> a tokenized training example
    with the prompt portion (system + user message, and the chat template's
    assistant-turn preamble) masked out of `labels` (-100) so the SFT/KD
    loss is only computed over the assistant's completion — the
    <tool_call>...</tool_call> block itself.

    The system prompt is built from harness.executor.build_system_prompt()
    over a registry containing ONLY this record's tool (not the whole
    8-tool registry) — matching the real dataset's per-example system
    prompts, which typically describe only the function(s) relevant to that
    example, and reusing the exact prompt-construction code path the
    harness itself uses at eval time (harness/tasks.py's
    source="glaive_train"/"glaive_test" tasks run through the same
    executor.build_system_prompt(), so training and eval never see two
    different tool-description formats for the same tool).
    """
    from adbench.harness.executor import build_system_prompt
    from adbench.harness.tools import ToolRegistry

    tool_name = record["expected_tool_sequence"][0]
    mini_registry = ToolRegistry()
    mini_registry.register(tool_registry.get(tool_name))
    system_prompt = build_system_prompt(mini_registry)

    user_message = record["user_goal"]
    assistant_completion = json.dumps(
        {"name": tool_name, "arguments": record["arguments"]}
    )
    assistant_completion = f"<tool_call>{assistant_completion}</tool_call>"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": assistant_completion}],
        tokenize=False, add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"Tokenized prompt is not a prefix of the tokenized full example for "
            f"{record.get('task_id', '<unknown>')!r} — the chat template likely "
            "changes earlier tokens when the assistant turn is appended (rather "
            "than only appending new ones), so prompt-length label masking isn't "
            "valid here. Inspect this record's formatting before training on it."
        )

    labels = list(full_ids)
    for i in range(len(prompt_ids)):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def compute_total_optimizer_steps(
    n_examples: int, batch_size: int, gradient_accumulation_steps: int, num_train_epochs: float
) -> int:
    """Number of *optimizer* steps (not batches) the full run will take —
    used to configure the LR scheduler. Easy to get wrong by forgetting the
    gradient-accumulation division (an optimizer step only happens once
    every `gradient_accumulation_steps` batches — see run_training_loop),
    which would silently under-run the schedule (e.g. a cosine decay never
    reaching its floor because the scheduler thinks there are more steps
    left than the run will actually take) — pulled out as its own pure,
    directly-tested function specifically because that bug is easy to miss
    just by reading the training loop."""
    batches_per_epoch = max(1, n_examples // batch_size)
    optimizer_steps_per_epoch = max(1, batches_per_epoch // gradient_accumulation_steps)
    return max(1, int(optimizer_steps_per_epoch * num_train_epochs))


def _pad_batch(examples: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, list[list[int]]]:
    """Right-pad a list of format_training_example() outputs to the batch's
    longest sequence. input_ids/attention_mask pad with pad_token_id/0;
    labels pad with -100 (so padding never contributes to the loss)."""
    max_len = max(len(ex["input_ids"]) for ex in examples)

    def pad_row(row: list[int], pad_value: int) -> list[int]:
        return row + [pad_value] * (max_len - len(row))

    return {
        "input_ids": [pad_row(ex["input_ids"], pad_token_id) for ex in examples],
        "attention_mask": [pad_row(ex["attention_mask"], 0) for ex in examples],
        "labels": [pad_row(ex["labels"], -100) for ex in examples],
    }


# --------------------------------------------------------------------------
# Loss-curve logging — plain JSONL, reusing adbench.data.prepare's writer so
# there's one write_jsonl implementation in the whole package.
# --------------------------------------------------------------------------

def loss_log_path(condition: str, sweep_name: str | None = None) -> Path:
    name = condition if sweep_name is None else f"{condition}-{sweep_name}"
    return REPO_ROOT / "results" / "training_logs" / f"{name}.jsonl"


def write_loss_log(entries: list[dict[str, Any]], path: str | Path) -> None:
    from adbench.data.prepare import write_jsonl
    write_jsonl(entries, path)


# --------------------------------------------------------------------------
# Orchestration — needs Unsloth/torch/a GPU for a real run. Not locally
# testable end-to-end; built from the tested pieces above.
# --------------------------------------------------------------------------

def load_student(models_config: dict[str, Any]):
    """Load the student model + tokenizer via Unsloth, wrapped with the LoRA
    config from configs/models.yaml. Returns (model, tokenizer)."""
    from unsloth import FastLanguageModel

    student_cfg = models_config["student"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=student_cfg["hf_id"],
        max_seq_length=student_cfg["max_seq_length"],
        load_in_4bit=student_cfg["load_in_4bit"],
    )
    lora_cfg = student_cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    )
    return model, tokenizer


def load_teacher(models_config: dict[str, Any]):
    """Load the teacher model (inference-only — no LoRA, no gradients) via
    Unsloth. Only called when a condition's kd_weight > 0."""
    from unsloth import FastLanguageModel

    teacher_cfg = models_config["teacher"]
    model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=teacher_cfg["hf_id"],
        max_seq_length=teacher_cfg["max_seq_length"],
        load_in_4bit=teacher_cfg["load_in_4bit"],
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def save_checkpoint(model, tokenizer, checkpoint_dir: Path) -> None:
    """Save the LoRA adapter (+ tokenizer) for this condition.

    Called for EVERY condition, including "base" — immediately after
    loading, with zero training steps. A freshly-initialized LoRA adapter's
    B matrix is zero-initialized (standard LoRA init), so its output delta
    is exactly zero until trained: saving it now means "base" produces a
    checkpoint that evaluation/run_eval.py can load through the exact same
    "base model + adapter from checkpoint" code path as sft_only/distilled,
    rather than needing a special case for "no adapter, load hf_id
    directly" — one loading path for all three conditions.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))


def run_training_loop(student, teacher, examples: list[dict[str, list[int]]], cfg: TrainingConfig, pad_token_id: int):
    """The actual multi-epoch training loop: batch, forward (student, and
    teacher if cfg.kd.kd_weight > 0), compute loss via training_step,
    backward, step the optimizer/scheduler, log every cfg.logging_steps,
    save a checkpoint every cfg.save_steps. Needs a real GPU-resident model
    to do anything meaningful — not exercised by the local test suite (see
    module docstring); training_step() and _pad_batch() above, which this
    function is built from, ARE unit-tested.

    Returns the list of logged {step, epoch, sft_loss, kd_loss, total_loss,
    learning_rate} dicts (also written to results/training_logs/ by
    train_condition()).
    """
    import torch
    from torch.optim import AdamW
    from transformers import get_scheduler

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    optimizer = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    batch_size = cfg.per_device_train_batch_size
    total_steps = compute_total_optimizer_steps(
        len(examples), batch_size, cfg.gradient_accumulation_steps, cfg.num_train_epochs
    )
    scheduler = get_scheduler(
        cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )

    log: list[dict[str, Any]] = []
    step = 0
    accum = 0
    optimizer.zero_grad()

    # Note: if an epoch's batch count isn't a multiple of
    # gradient_accumulation_steps, the leftover accumulated gradients carry
    # over into the next epoch's first few batches rather than being
    # flushed at the epoch boundary — a minor, accepted simplification (one
    # optimizer step occasionally mixes two epochs' batches at the seam),
    # not a correctness concern given how small that fraction of steps is.
    for epoch in range(int(cfg.num_train_epochs)):
        order = list(range(len(examples)))
        random.shuffle(order)
        for batch_start in range(0, len(order) - batch_size + 1, batch_size):
            batch_indices = order[batch_start : batch_start + batch_size]
            batch = _pad_batch([examples[i] for i in batch_indices], pad_token_id)
            device = next(student.parameters()).device
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)

            loss, components = training_step(student, teacher, input_ids, attention_mask, labels, cfg.kd)
            (loss / cfg.gradient_accumulation_steps).backward()
            accum += 1

            if accum == cfg.gradient_accumulation_steps:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum = 0
                step += 1

                if step % cfg.logging_steps == 0:
                    log.append({
                        "step": step, "epoch": epoch,
                        "sft_loss": components["sft_loss"], "kd_loss": components["kd_loss"],
                        "total_loss": components["total_loss"],
                        "learning_rate": scheduler.get_last_lr()[0],
                    })
                if step % cfg.save_steps == 0:
                    # Adapter-only intermediate save (cheap, frequent); the
                    # tokenizer is saved once at the end by
                    # save_checkpoint() in train_condition() — it never
                    # changes during training, no need to rewrite it every
                    # cfg.save_steps.
                    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    student.save_pretrained(str(cfg.checkpoint_dir))

    return log


def train_condition(
    condition: str,
    experiment_config: dict[str, Any],
    models_config: dict[str, Any],
    sweep_name: str | None = None,
    glaive_train_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one condition end to end: load model(s), (for sft_only/distilled)
    train on the glaive train split, save the checkpoint. Returns a small
    summary dict. See module docstring — needs Colab (Unsloth + GPU)."""
    from adbench.data.prepare import read_jsonl
    from adbench.harness.tools import build_glaive_registry

    cfg = resolve_training_config(experiment_config, condition, sweep_name)
    student, tokenizer = load_student(models_config)

    if condition == "base":
        save_checkpoint(student, tokenizer, cfg.checkpoint_dir)
        return {"condition": condition, "checkpoint_dir": str(cfg.checkpoint_dir), "steps": 0}

    teacher = load_teacher(models_config) if cfg.kd.kd_weight > 0 else None

    if glaive_train_path is None:
        data_config = load_experiment_config(REPO_ROOT / "configs" / "data.yaml")
        glaive_train_path = REPO_ROOT / data_config["output"]["train_path"]
    records = read_jsonl(glaive_train_path)

    registry = build_glaive_registry()
    examples = [format_training_example(r, tokenizer, registry) for r in records]

    log = run_training_loop(student, teacher, examples, cfg, tokenizer.pad_token_id)

    save_checkpoint(student, tokenizer, cfg.checkpoint_dir)
    write_loss_log(log, loss_log_path(condition, sweep_name))

    return {
        "condition": condition, "checkpoint_dir": str(cfg.checkpoint_dir),
        "steps": len(log), "n_examples": len(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--sweep", default=None, help="Named entry in training.sweep (configs/experiment.yaml).")
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.experiment_config)
    models_config = load_experiment_config(args.models_config)

    summary = train_condition(args.condition, experiment_config, models_config, sweep_name=args.sweep)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
