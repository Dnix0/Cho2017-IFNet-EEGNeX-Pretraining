# Copyright (c) 2026 Project contributors
# SPDX-License-Identifier: Apache-2.0

"""Pre-specified probability-level IFNet + EEGNeX ensemble."""

import torch


def fixed_probability_ensemble(ifnet_logits, eegnex_logits):
    if ifnet_logits.shape != eegnex_logits.shape:
        raise ValueError(
            f"Logit shape mismatch: {ifnet_logits.shape} vs "
            f"{eegnex_logits.shape}"
        )
    p_ifnet = torch.softmax(ifnet_logits, dim=1)
    p_eegnex = torch.softmax(eegnex_logits, dim=1)
    probabilities = 0.75 * p_ifnet + 0.25 * p_eegnex
    if not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(probabilities.shape[0], device=probabilities.device),
        atol=1e-6,
    ):
        raise RuntimeError("Ensemble probability sums are invalid.")
    return probabilities
