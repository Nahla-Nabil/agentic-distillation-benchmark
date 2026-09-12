"""Training for all three conditions. Runs in Colab against a free T4 via
Unsloth (see requirements-colab.txt and notebooks/02_training.ipynb, which
drives this package).

    train.py  -> the single script for all three conditions, selected via
                 --condition {base,sft_only,distilled}: base loads and saves
                 immediately (no training), sft_only and distilled run the
                 same training loop differing only in the loss's KD
                 weighting (see train.py::resolve_training_config).
    losses.py -> the combined KD+SFT loss both trained conditions share
                 (sft_only forces kd_weight=0.0, so the two conditions
                 differ only in this weighting, not in data, LoRA setup, or
                 other hyperparameters).
"""
