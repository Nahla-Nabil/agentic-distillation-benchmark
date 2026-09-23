"""evaluation/ablation_worker.py's own bits (job parsing, result_path, describe) — no GPU, no
network. Job running itself is tested once in tests/test_worker_utils.py and reused here as
evaluation/ext_worker.py and bfcl_worker.py do."""

from adbench.evaluation.ablation_worker import (
    describe_ablation_result,
    parse_ablation_jobs,
    result_path,
    run_jobs,
    split_jobs,
)


def test_parse_ablation_jobs():
    assert parse_ablation_jobs("0:2:sft_only, 3:4:distilled ,") == [(0, 2, "sft_only"), (3, 4, "distilled")]


def test_parse_ablation_jobs_rejects_the_wrong_shape():
    import pytest

    with pytest.raises(ValueError):
        parse_ablation_jobs("0:sft_only")           # missing n_tools
    with pytest.raises(ValueError):
        parse_ablation_jobs("0:2:4:sft_only")        # too many fields


def test_result_path_includes_n_tools_and_condition():
    assert result_path(3, 4, "distilled") == "runs/v2-seed3/results/stages/ablation_n4_distilled.json"
    assert result_path(3, 2, "distilled") != result_path(3, 4, "distilled")


def test_describe_ablation_result():
    rows = [{"success": True, "train_steps": 24}, {"success": False, "train_steps": 24}, {"success": True, "train_steps": 24}]
    assert describe_ablation_result(rows) == "24 train steps, 3 eval tasks, success 0.667"


def test_describe_ablation_result_empty():
    assert describe_ablation_result([]) == "? train steps, 0 eval tasks, success 0.000"


def test_reexported_job_helpers_are_the_shared_ones():
    from adbench.evaluation import worker_utils

    assert split_jobs is worker_utils.split_jobs
    assert run_jobs is worker_utils.run_jobs


def test_the_seed_n_tools_key_folding_round_trips():
    """The way main() folds (n_tools, condition) into worker_utils' generic (seed, key) job
    shape and splits it back — the trick the module docstring/comment describes."""
    key = f"{4}:{'distilled'}"
    n_tools_str, condition = key.split(":", 1)
    assert (int(n_tools_str), condition) == (4, "distilled")
    # condition names never themselves contain ":", so this is always unambiguous
    from adbench.training.train import ALL_CONDITIONS

    assert all(":" not in c for c in ALL_CONDITIONS)
