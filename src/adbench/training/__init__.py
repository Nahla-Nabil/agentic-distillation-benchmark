"""Training scripts for the two fine-tuned conditions. Both run in Colab
against a free T4 via Unsloth (see requirements-colab.txt and the
notebooks/02_*, 03_* notebooks that drive these modules).

    sft_train.py     -> condition 2: SFT-only student (control)
    distill_train.py -> condition 3: KD + SFT student (treatment)
    losses.py        -> the combined KD+SFT loss shared conceptually by both
                         (SFT-only just runs with the KD term's weight at 0,
                         so the two conditions differ only in loss config,
                         not in data, LoRA setup, or hyperparameters).
"""
