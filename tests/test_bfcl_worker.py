"""evaluation/bfcl_worker.py's own bits (result_path, describe_bfcl_result) — no GPU, no network.
Job parsing/splitting/running itself is tested once in tests/test_worker_utils.py and re-used
here via import, matching ext_worker.py's pattern (tests/test_ext_worker.py)."""

from adbench.evaluation.bfcl_worker import describe_bfcl_result, parse_jobs, result_path, run_jobs, split_jobs


def test_result_path_matches_the_tag_layout_used_by_the_pipeline():
    assert result_path(3, "distilled") == "runs/v2-seed3/results/stages/bfcl_eval_simple_distilled.json"
    assert result_path(3, "distilled", "multiple") == "runs/v2-seed3/results/stages/bfcl_eval_multiple_distilled.json"


def test_describe_bfcl_result():
    rows = [{"success": True}, {"success": True}, {"success": False}, {"success": True}]
    assert describe_bfcl_result(rows) == "4 examples, success 0.750"
    assert describe_bfcl_result([]) == "0 examples, success 0.000"


def test_reexported_job_helpers_are_the_shared_ones():
    # ext_worker.py and bfcl_worker.py must not silently drift into two copies of this logic
    from adbench.evaluation import worker_utils

    assert parse_jobs is worker_utils.parse_jobs
    assert split_jobs is worker_utils.split_jobs
    assert run_jobs is worker_utils.run_jobs
