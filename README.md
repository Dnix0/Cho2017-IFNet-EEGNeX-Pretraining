# Cho2017 IFNet / EEGNeX Pretraining Checkpoints

> Channel-specific checkpoint release. Checkpoint tracks are not interchangeable.

## Released tracks

### Common10

- Channels: `FC3, FC4, C5, C3, C1, C2, C4, C6, CP3, CP4`
- Input shape: `(batch, 10, 750)`
- Models: IFNet and EEGNeX
- Checkpoints are Cho2017 source-only seed42 best checkpoints.

### Common6

- Channels: `FC3, FC4, C3, C4, CP3, CP4`
- Input shape: `(batch, 6, 750)`
- Model: IFNet
- Checkpoint is the Cho2017 source-only seed42 best checkpoint.

### Common8

- Not included in this release. No verified Cho2017 source-only Common8 checkpoint was found.
- HS downstream/fine-tuned checkpoints are not substitutes for a Cho2017 source-only Common8 checkpoint.

## Legacy full22

- The prior 22-channel IFNet and EEGNeX resources remain legacy tracks.
- Do not use a full22 checkpoint with Common10 or Common6 inputs.

## Selection rule

1. Verify the downstream dataset channel names from an official source.
2. Use the exact channel count, names, and order for the selected track.
3. Verify SHA-256 using `manifests/channel_lineage_manifest.json`.
4. Use strict checkpoint loading.
5. Replace the classifier head when downstream class semantics differ.

## Safety and scope

- No patient EEG, FMA scores, downstream OOF features, or patient-specific checkpoints are included.
- Liu2024 LOSO checkpoints and HS fine-tuned checkpoints are excluded.
