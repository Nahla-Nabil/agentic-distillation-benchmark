"""evaluation/teacher_eval_worker.py's own bits - no GPU, no network."""

from adbench.evaluation.teacher_eval_worker import describe_teacher_result, result_path
from adbench.training.train import load_experiment_config


def test_result_path_is_namespaced_by_models_yaml_key():
    assert result_path(0, "teacher") == "runs/v2-seed0/results/stages/teacher_eval_teacher.json"
    assert result_path(0, "teacher_olmo13").endswith("teacher_eval_teacher_olmo13.json")


def test_every_teacher_key_it_will_be_given_exists_in_models_yaml():
    models = load_experiment_config("configs/models.yaml")
    for key in ("teacher", "teacher_8b", "teacher_olmo", "teacher_olmo13"):
        assert models[key]["role"] == "teacher"


def test_describe_teacher_result():
    assert describe_teacher_result([{"success": True}, {"success": False}]) == "2 eval tasks, success 0.500"
