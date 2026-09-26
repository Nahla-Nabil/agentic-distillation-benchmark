"""evaluation/sweep_worker.py's own bits (job parsing, paths, key folding) - no GPU, no network."""

import pytest

from adbench.evaluation.sweep_worker import (
    describe_sweep_result,
    fold_key,
    loss_log_repo_path,
    parse_sweep_jobs,
    result_path,
    split_key,
)
from adbench.training.train import load_experiment_config, resolve_training_config


def test_parse_sweep_jobs():
    assert parse_sweep_jobs("0:main:lower_lr:sft_only, 2:small:sft_heavy:self_distill_small ,") == [
        (0, "main", "lower_lr", "sft_only"), (2, "small", "sft_heavy", "self_distill_small"),
    ]


def test_parse_sweep_jobs_rejects_bad_shapes_and_unknown_pairs():
    with pytest.raises(ValueError):
        parse_sweep_jobs("0:main:sft_only")
    with pytest.raises(ValueError):
        parse_sweep_jobs("0:huge:lower_lr:sft_only")


def test_result_and_log_paths_include_every_varying_dimension():
    a = result_path(1, "main", "lower_lr", "sft_only")
    assert a == "runs/v2-seed1/results/stages/sweep_main_lower_lr_sft_only.json"
    others = {
        a,
        result_path(1, "small", "lower_lr", "sft_only"),
        result_path(1, "main", "lower_lr_1ep", "sft_only"),
        result_path(1, "main", "lower_lr", "distilled"),
        result_path(2, "main", "lower_lr", "sft_only"),
    }
    assert len(others) == 5
    assert loss_log_repo_path(1, "main", "lower_lr", "sft_only").endswith("sweep_main_lower_lr_sft_only.jsonl")


def test_key_folding_round_trips():
    key = fold_key("small", "sft_heavy", "self_distill_small")
    assert split_key(key) == ("small", "sft_heavy", "self_distill_small")


def test_describe_sweep_result():
    rows = [{"success": True, "train_steps": 24}, {"success": False, "train_steps": 24}]
    assert describe_sweep_result(rows) == "24 train steps, 2 eval tasks, success 0.500"


def test_the_sweeps_this_worker_relies_on_exist_and_do_what_they_say():
    cfg = load_experiment_config("configs/experiment.yaml")

    def g(cond, sweep):
        return resolve_training_config(cfg, cond, sweep)

    assert g("sft_only", "lower_lr_1ep").learning_rate == 5e-5
    assert g("sft_only", "lower_lr_1ep").num_train_epochs == 1
    assert g("sft_only", "lower_lr_1ep").kd.kd_weight == 0.0  # sft_only never uses KD
    vh = g("self_distill_small", "sft_vheavy").kd
    assert (vh.kd_weight, vh.sft_weight) == (0.05, 0.95)
    assert g("self_distill", "sft_heavy").kd.kd_weight == 0.2
    assert g("self_distill", "kd_heavy").kd.kd_weight == 0.8
    assert g("self_distill_small", "kd_0p1").kd.kd_weight == 0.1
    assert g("self_distill", None).kd.kd_weight == 0.5  # baseline untouched by the additions


def test_split_sweep_spec():
    from adbench.evaluation.sweep_worker import split_sweep_spec

    assert split_sweep_spec("lower_lr") == ("lower_lr", None)
    assert split_sweep_spec("baseline@n20") == ("baseline", 20)
    assert split_sweep_spec("sft_heavy@n40") == ("sft_heavy", 40)
    for bad in ("baseline@n", "baseline@nx", "baseline@n0"):
        with pytest.raises(ValueError):
            split_sweep_spec(bad)


def test_data_scale_results_never_collide_with_full_data_results():
    assert result_path(0, "main", "baseline@n20", "sft_only") != result_path(0, "main", "baseline", "sft_only")
    assert result_path(0, "main", "baseline@n20", "sft_only") != result_path(0, "main", "baseline@n40", "sft_only")
    assert parse_sweep_jobs("0:main:baseline@n20:sft_only") == [(0, "main", "baseline@n20", "sft_only")]
