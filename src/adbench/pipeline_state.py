"""Resumable pipeline runs: skip stages that already finished, and keep their
outputs outside the (disposable) notebook session so a later run — on
another Kaggle session, another account, or another machine — picks up where
the last one stopped instead of starting from zero.

A stage is "done" when results/stages/<stage>.done exists. finish(stage)
writes that marker and pushes the persisted directories (checkpoints,
results, data splits) to a private Hugging Face repo; restore() pulls them
back at the start of a run.

Persistence is opt-in through environment variables, and the pipeline works
without it (stages still skip within a session; nothing survives a lost one):

    HF_TOKEN            a Hugging Face token with write access (on Kaggle,
                        keep it in Notebook -> Add-ons -> Secrets, never in
                        the notebook text)
    ADBENCH_HF_REPO     "<user>/<repo>", created private if it doesn't exist
    ADBENCH_RUN_TAG     namespace inside the repo (default "v1"); change it
                        after a code change that invalidates earlier outputs
                        (e.g. a training fix) so stale checkpoints are
                        never restored into a new run.

A failed upload or download only prints a warning — losing persistence must
never crash a run that is otherwise fine.
"""

import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Everything a later stage or the final analysis needs. Relative to REPO_ROOT.
PERSISTED_DIRS = ("checkpoints", "results", "data/splits", "data/general_eval")


def stages_dir() -> Path:
    return REPO_ROOT / "results" / "stages"


def stage_marker(stage: str) -> Path:
    return stages_dir() / f"{stage}.done"


def is_done(stage: str) -> bool:
    return stage_marker(stage).exists()


def _settings() -> tuple[str, str, str] | None:
    repo = os.environ.get("ADBENCH_HF_REPO")
    token = os.environ.get("HF_TOKEN")
    if not repo or not token:
        return None
    return repo, token, os.environ.get("ADBENCH_RUN_TAG", "v1")


def persistence_enabled() -> bool:
    return _settings() is not None


# Thin wrappers so tests can stub the network without importing huggingface_hub.
def _hf_api(token: str):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def _snapshot_download(**kwargs):
    from huggingface_hub import snapshot_download
    return snapshot_download(**kwargs)


def restore() -> bool:
    """Pull the previous run's outputs into REPO_ROOT. Call once, at the very
    start, before any stage runs. Returns True if something was restored."""
    settings = _settings()
    if settings is None:
        print("[persist] OFF (set HF_TOKEN and ADBENCH_HF_REPO to resume across sessions).")
        return False
    repo, token, tag = settings
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _snapshot_download(
                repo_id=repo, repo_type="model", token=token, local_dir=tmp,
                allow_patterns=[f"runs/{tag}/**"],
            )
            source = Path(tmp) / "runs" / tag
            if not source.exists():
                print(f"[persist] nothing to restore from {repo} (tag {tag!r}) — starting fresh.")
                return False
            shutil.copytree(source, REPO_ROOT, dirs_exist_ok=True)
    except Exception as e:  # noqa: BLE001 — a missing repo/network must not stop the run
        print(f"[persist] could not restore from {repo}: {type(e).__name__}: {e}. Starting fresh.")
        return False
    done = sorted(p.stem for p in stages_dir().glob("*.done"))
    print(f"[persist] restored from {repo} (tag {tag!r}); finished stages: {done}")
    return True


def check_upload() -> None:
    """Fail fast if persistence is configured but cannot actually write — a
    bad token or a read-only one would otherwise only show up as a warning
    after hours of work that then cannot be resumed. No-op when persistence
    is off."""
    settings = _settings()
    if settings is None:
        return
    repo, token, tag = settings
    try:
        api = _hf_api(token)
        api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=b"ok", path_in_repo=f"runs/{tag}/results/.write_check",
            repo_id=repo, repo_type="model", commit_message="startup write check",
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"HF_TOKEN is set but writing to {repo!r} failed ({type(e).__name__}: {e}). "
            "Use a token with Write access, or remove the HF_TOKEN secret to run without resume."
        ) from e
    print(f"[persist] ON — write access to {repo} confirmed (tag {tag!r}).")


def push(message: str) -> bool:
    """Upload the persisted directories. Returns False (after a warning) on
    any failure, or when persistence is off."""
    settings = _settings()
    if settings is None:
        return False
    repo, token, tag = settings
    try:
        api = _hf_api(token)
        api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
        for rel in PERSISTED_DIRS:
            folder = REPO_ROOT / rel
            if folder.exists():
                api.upload_folder(
                    folder_path=str(folder), path_in_repo=f"runs/{tag}/{rel}",
                    repo_id=repo, repo_type="model", commit_message=message,
                    ignore_patterns=["*.tmp", "__pycache__/*"],
                )
    except Exception as e:  # noqa: BLE001
        print(f"[persist] upload failed ({message}): {type(e).__name__}: {e}. Results stay local only.")
        return False
    print(f"[persist] saved to {repo} ({message})")
    return True


def finish(stage: str) -> None:
    """Mark a stage done and persist everything produced so far."""
    stages_dir().mkdir(parents=True, exist_ok=True)
    stage_marker(stage).write_text(json.dumps({"stage": stage, "finished_at": time.time()}))
    push(f"stage {stage}")


def run_stage(stage: str, fn: Callable[[], None]) -> bool:
    """Run fn() unless the stage already finished. Returns True if it ran.
    The marker is written only after fn() returns, so an exception leaves the
    stage pending and a re-run redoes it."""
    if is_done(stage):
        print(f"[stage] {stage}: already done — skipping.")
        return False
    print(f"[stage] {stage}: running ...")
    fn()
    finish(stage)
    return True
