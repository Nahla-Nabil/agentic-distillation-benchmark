"""Tests for pipeline_state.py — stage skipping and the persist/restore
round trip, with the Hugging Face calls stubbed (no network)."""

import pytest

from adbench import pipeline_state as ps


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "REPO_ROOT", tmp_path)
    for var in ("HF_TOKEN", "ADBENCH_HF_REPO", "ADBENCH_RUN_TAG"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


class _FakeApi:
    def __init__(self, store):
        self.store = store  # {(repo, path_in_repo): sorted file names}
        self.created = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_folder(self, folder_path, path_in_repo, repo_id, **kwargs):
        from pathlib import Path
        root = Path(folder_path)
        for f in root.rglob("*"):
            if f.is_file():
                self.store[(repo_id, f"{path_in_repo}/{f.relative_to(root).as_posix()}")] = f.read_bytes()


def _enable_persistence(monkeypatch, store, tag="v1"):
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setenv("ADBENCH_HF_REPO", "user/repo")
    monkeypatch.setenv("ADBENCH_RUN_TAG", tag)
    api = _FakeApi(store)
    monkeypatch.setattr(ps, "_hf_api", lambda token: api)

    def fake_snapshot(repo_id, local_dir, allow_patterns, **kwargs):
        from pathlib import Path
        prefix = allow_patterns[0].removesuffix("/**")
        for (repo, path), data in store.items():
            if repo == repo_id and path.startswith(prefix + "/"):
                target = Path(local_dir) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)

    monkeypatch.setattr(ps, "_snapshot_download", fake_snapshot)
    return api


def test_run_stage_runs_once_then_skips(root):
    calls = []
    assert ps.run_stage("train_x", lambda: calls.append(1)) is True
    assert ps.run_stage("train_x", lambda: calls.append(1)) is False
    assert calls == [1]
    assert ps.is_done("train_x")


def test_failed_stage_is_not_marked_done(root):
    def boom():
        raise RuntimeError("OOM")

    with pytest.raises(RuntimeError):
        ps.run_stage("train_x", boom)
    assert not ps.is_done("train_x")
    assert ps.run_stage("train_x", lambda: None) is True  # a re-run redoes it


def test_persistence_off_without_env(root, capsys):
    assert ps.persistence_enabled() is False
    assert ps.restore() is False
    assert ps.push("x") is False
    ps.run_stage("s", lambda: None)  # still works, locally
    assert ps.is_done("s")


def test_finish_pushes_and_a_fresh_root_restores_it(root, monkeypatch, tmp_path_factory):
    store = {}
    _enable_persistence(monkeypatch, store)

    (root / "checkpoints" / "sft_only").mkdir(parents=True)
    (root / "checkpoints" / "sft_only" / "adapter_model.safetensors").write_bytes(b"weights")
    (root / "results").mkdir()
    (root / "results" / "eval_summary.json").write_text("{}")
    ps.run_stage("train_sft_only", lambda: None)

    # A brand-new container: empty repo dir, same HF repo.
    fresh = tmp_path_factory.mktemp("fresh")
    monkeypatch.setattr(ps, "REPO_ROOT", fresh)
    assert ps.restore() is True
    assert (fresh / "checkpoints" / "sft_only" / "adapter_model.safetensors").read_bytes() == b"weights"
    assert ps.is_done("train_sft_only")
    calls = []
    assert ps.run_stage("train_sft_only", lambda: calls.append(1)) is False
    assert calls == []


def test_run_tag_isolates_runs(root, monkeypatch, tmp_path_factory):
    store = {}
    _enable_persistence(monkeypatch, store, tag="v1")
    (root / "results").mkdir()
    ps.run_stage("old_stage", lambda: None)

    fresh = tmp_path_factory.mktemp("fresh")
    monkeypatch.setattr(ps, "REPO_ROOT", fresh)
    monkeypatch.setenv("ADBENCH_RUN_TAG", "v2")  # e.g. after a training fix
    assert ps.restore() is False
    assert not ps.is_done("old_stage")


def test_upload_failure_only_warns(root, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setenv("ADBENCH_HF_REPO", "user/repo")

    def broken(token):
        raise ConnectionError("offline")

    monkeypatch.setattr(ps, "_hf_api", broken)
    (root / "results").mkdir()
    ps.run_stage("s", lambda: None)  # must not raise
    assert ps.is_done("s")
    assert "upload failed" in capsys.readouterr().out
