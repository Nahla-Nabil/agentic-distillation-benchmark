"""evaluation/third_pair_worker.py's own bits (result_path, describe, constants) - no GPU, no
network. Job parsing/running is tested once in tests/test_worker_utils.py and reused here."""

from adbench.evaluation.third_pair_worker import (
    STUDENT_KEY,
    TASK_SET,
    describe_pair3_result,
    loss_log_repo_path,
    parse_jobs,
    result_path,
    run_jobs,
    split_jobs,
)
from adbench.training.train import load_experiment_config, resolve_training_config


def test_result_path_matches_the_tag_layout_used_by_the_pipeline():
    assert result_path(2, "sft_only") == "runs/v2-seed2/results/stages/pair3_sft_only.json"
    assert result_path(0, "base") == "runs/v2-seed0/results/stages/pair3_base.json"
    assert result_path(1, "distilled_olmo") == "runs/v2-seed1/results/stages/pair3_distilled_olmo.json"


def test_result_path_never_collides_with_the_other_two_pairs():
    from adbench.evaluation.second_pair_worker import result_path as pair2_result_path

    for seed, cond in ((0, "sft_only"), (0, "base")):
        assert result_path(seed, cond) != pair2_result_path(seed, cond)


def test_loss_log_repo_path_is_namespaced_away_from_the_other_pairs_logs():
    assert loss_log_repo_path(1, "distilled_olmo") == "runs/v2-seed1/results/training_logs/pair3_distilled_olmo.jsonl"


def test_describe_pair3_result():
    rows = [{"success": True, "train_steps": 24}, {"success": False, "train_steps": 24}]
    assert describe_pair3_result(rows) == "24 train steps, 2 eval tasks, success 0.500"
    assert describe_pair3_result([]) == "? train steps, 0 eval tasks, success 0.000"


def test_uses_the_olmo_student_and_the_shared_unseen_tools_task_set():
    assert STUDENT_KEY == "student_olmo"
    assert TASK_SET == "unseen_tools"


def test_reexported_job_helpers_are_the_shared_ones():
    from adbench.evaluation import worker_utils

    assert parse_jobs is worker_utils.parse_jobs
    assert split_jobs is worker_utils.split_jobs
    assert run_jobs is worker_utils.run_jobs


def test_the_conditions_this_worker_relies_on_exist_and_point_at_this_pairs_own_teachers():
    cfg = load_experiment_config("configs/experiment.yaml")
    assert resolve_training_config(cfg, "sft_only").kd.kd_weight == 0.0
    distilled = resolve_training_config(cfg, "distilled_olmo")
    assert (distilled.kd.kd_weight, distilled.teacher_key) == (0.5, "teacher_olmo")
    self_distill = resolve_training_config(cfg, "self_distill_olmo")
    assert (self_distill.kd.kd_weight, self_distill.teacher_key) == (0.5, "teacher_self_olmo")


def test_models_config_has_this_pairs_three_entries_with_matching_lora_setup():
    models = load_experiment_config("configs/models.yaml")
    for key in ("teacher_olmo", "teacher_self_olmo", "student_olmo"):
        assert key in models, key
    student = models["student_olmo"]
    assert student["hf_id"] == "allenai/OLMo-2-0425-1B-Instruct"
    assert models["teacher_self_olmo"]["hf_id"] == student["hf_id"]  # self-teacher = the student's own base
    assert models["teacher_olmo"]["hf_id"] != student["hf_id"]
    assert student["lora"]["r"] == 16
