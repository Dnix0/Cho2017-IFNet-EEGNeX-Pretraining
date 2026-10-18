# Copyright (c) 2026 Project contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading helpers for downstream patient-data fine-tuning."""

from pathlib import Path

import torch

from src.models.eegnex_2class import EEGNeXStyleBackbone
from src.models.ifnet_2class import IFNetBackbone


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_cho2017_pretrained(
    model_name,
    checkpoint_path,
    device="cpu",
    strict=True,
):
    model_name = model_name.lower()
    if model_name == "ifnet":
        model_class = IFNetBackbone
    elif model_name == "eegnex":
        model_class = EEGNeXStyleBackbone
    else:
        raise ValueError("model_name must be 'ifnet' or 'eegnex'.")

    checkpoint_path = Path(checkpoint_path)
    checkpoint = _torch_load(checkpoint_path)
    metadata = checkpoint.get("checkpoint_metadata", {})

    if metadata.get("mode") != "full_source_pretraining":
        raise RuntimeError(
            "Checkpoint is not marked as a full-source pretraining artifact."
        )
    for key, expected in (
        ("dataset", "Cho2017"),
        ("n_classes", 2),
        ("n_channels", 22),
        ("n_samples", 750),
    ):
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"Checkpoint metadata mismatch for {key}: "
                f"{metadata.get(key)} != {expected}"
            )

    model = model_class(n_classes=2, n_channels=22)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.to(device)
    return model, metadata, checkpoint


def replace_classifier_for_new_task(model, n_output_classes):
    """Replace the final Linear layer for a downstream task.

    This mutates and returns the model. The downstream team must verify that
    replacing only the head is appropriate for the patient-data task.
    """
    linear_names = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    if not linear_names:
        raise RuntimeError("No Linear layer found.")

    final_name = linear_names[-1]
    parent = model
    parts = final_name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    old_layer = parent[int(parts[-1])] if parts[-1].isdigit() else getattr(parent, parts[-1])
    new_layer = torch.nn.Linear(old_layer.in_features, int(n_output_classes))
    if parts[-1].isdigit():
        parent[int(parts[-1])] = new_layer
    else:
        setattr(parent, parts[-1], new_layer)
    return model
