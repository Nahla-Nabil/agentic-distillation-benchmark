"""evaluation/second_pair_worker.py's own bits (result_path, describe, constants) — no GPU, no
network. Job parsing/running is tested once in tests/test_worker_utils.py and reused here."""

from adbench.evaluation.second_pair_worker import (
    STUDENT_KEY,
    TASK_SET,
    describe_pair2_result,
    parse_jobs,
    result_path,
    run_jobs,
    split_jobs,
)


def test_result_path_matches_the_tag_layout_used_by_the_pipeline():
    assert result_path(2, "distilled_8b") == "runs/v2-seed2/results/stages/pair2_distilled_8b.json"


def test_describe_pair2_result():
    rows = [{"success": True, "train_steps": 240}, {"success": False, "train_steps": 240}]
    assert describe_pair2_result(rows) == "240 train steps, 2 eval tasks, success 0.500"
    assert describe_pair2_result([]) == "? train steps, 0 eval tasks, success 0.000"


def test_uses_the_small_student_and_the_shared_unseen_tools_task_set():
    assert STUDENT_KEY == "student_small"
    assert TASK_SET == "unseen_tools"


def test_reexported_job_helpers_are_the_shared_ones():
    from adbench.evaluation import worker_utils

    assert parse_jobs is worker_utils.parse_jobs
    assert split_jobs is worker_utils.split_jobs
    assert run_jobs is worker_utils.run_jobs


def test_summarize_loss_log_reports_first_last_and_nonfinite():
    from adbench.evaluation.second_pair_worker import summarize_loss_log

    entries = [
        {"step": 10, "sft_loss": 2.0, "kd_loss": 1.5},
        {"step": 20, "sft_loss": float("nan"), "kd_loss": 1.0},
        {"step": 30, "sft_loss": 0.5, "kd_loss": 0.8},
    ]
    s = summarize_loss_log(entries)
    assert (s["n_logged"], s["sft_first"], s["sft_last"], s["kd_first"], s["kd_last"]) == (3, 2.0, 0.5, 1.5, 0.8)
    assert s["nonfinite"] == 1
    assert summarize_loss_log([])["sft_first"] is None


def test_loss_log_repo_path_is_namespaced_away_from_the_main_pipelines_logs():
    from adbench.evaluation.second_pair_worker import loss_log_repo_path

    assert loss_log_repo_path(1, "distilled_8b") == "runs/v2-seed1/results/training_logs/pair2_distilled_8b.jsonl"
