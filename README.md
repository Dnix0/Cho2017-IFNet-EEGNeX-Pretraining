# Cho2017 IFNet / EEGNeX 채널별 사전학습 및 HS Stroke 전이 결과

[한국어](#한국어) | [English](#english)

## 한국어

### 저장소 개요

이 저장소는 Cho2017 motor-imagery EEG로 학습한 IFNet 및 EEGNeX 체크포인트와 채널별 계보를 제공합니다. 체크포인트는 채널 수와 채널 순서가 정확히 일치할 때만 사용할 수 있으며, 서로 다른 계보를 임의로 교환하거나 누락 채널을 추정해서는 안 됩니다.

### 공개 체크포인트

#### Common10

- 채널 순서: `FC3, FC4, C5, C3, C1, C2, C4, C6, CP3, CP4`
- 입력 형태: `(batch, 10, 750)`
- 공개 모델: IFNet, EEGNeX
- 용도: 공식 Common10 채널 교집합을 갖는 downstream 데이터셋

#### Common6

- 채널 순서: `FC3, FC4, C3, C4, CP3, CP4`
- 입력 형태: `(batch, 6, 750)`
- 공개 모델: IFNet

#### Common8

- 이번 공개본에는 포함하지 않았습니다.
- 검증된 Cho2017 source-only Common8 체크포인트가 확인되지 않았으며, HS 환자 데이터로 학습된 downstream 체크포인트는 source-only 사전학습 체크포인트를 대체하지 않습니다.

#### Legacy full22

기존 22채널 IFNet·EEGNeX 체크포인트는 legacy 계보로 유지합니다. Full22 체크포인트를 Common10 또는 Common6 입력에 사용하면 안 됩니다.

### HS Stroke downstream 평가

Cho2017 사전학습 IFNet 표현을 14명의 종단 뇌졸중 MI-EEG 데이터에 적용했습니다. 모든 평가는 환자 단위 held-out OOF 방식으로 수행했으며, 동일 환자의 반복 세션이 서로 다른 fold에 섞이지 않도록 했습니다.

#### 분류 결과

- 독립 환자 수: 14명
- Pretrained IFNet 평균 Accuracy: 50.48%
- Pretrained IFNet 평균 Balanced Accuracy: 50.48%
- Pretrained IFNet 평균 Macro F1: 47.55%
- Pretrained IFNet 평균 class 0 recall: NA
- Pretrained IFNet 평균 class 1 recall: NA
- Random-frozen IFNet 평균 Balanced Accuracy: 50.03%
- Pretrained - Random-frozen Balanced Accuracy: 0.45%p
- 환자별 승/무/패: 9/1/4

`NA`로 표시된 지표는 원본 fold JSON에 일관된 집계값이 없어 공개 결과에서 제외한 항목입니다.

#### 임상 회귀용 표현 패키지

공개하지 않은 내부 협업 패키지에서는 다음 세 표현군을 동일 환자 정렬로 준비했습니다.

1. Cho2017-pretrained IFNet OOF learned feature
2. Random-frozen IFNet control feature
3. Physiological EEG feature baseline

IFNet assessment feature는 trial-level 512D feature의 전체 평균 512D와 class 1 - class 0 contrast 512D를 결합한 1,024 predictors로 구성됩니다.

- 독립 환자: 14명
- assessment-level: 28행
- patient-level change: 14행
- pretrained/random trial alignment: 통과
- pretrained/random assessment alignment: 통과

이 자료는 작은 표본의 feasibility 분석용입니다. Assessment 파일은 28행이지만 독립 표본은 환자 14명이므로, 무작위 행 분할이 아니라 `patient_id` 기준 LOSO를 사용해야 합니다. Scaling, PCA, feature selection, hyperparameter 선택은 training fold 내부에서만 수행해야 합니다.

### 공개 범위

이 저장소에는 환자 EEG, FMA 점수, 환자별 OOF feature·prediction, HS downstream checkpoint를 포함하지 않습니다. 공개하는 HS 결과는 집계 통계와 분석 프로토콜에 한정됩니다.

### 무결성

체크포인트 SHA-256과 내부 parameter shape는 `manifests/channel_lineage_manifest.json`에서 확인할 수 있습니다.

---
# English

## Overview

This repository provides channel-specific IFNet and EEGNeX checkpoints trained on Cho2017 motor-imagery EEG. A checkpoint may be used only when the downstream channel count, channel names, and channel order exactly match the corresponding lineage.

## Released checkpoints

### Common10

- Channel order: `FC3, FC4, C5, C3, C1, C2, C4, C6, CP3, CP4`
- Input shape: `(batch, 10, 750)`
- Models: IFNet and EEGNeX

### Common6

- Channel order: `FC3, FC4, C3, C4, CP3, CP4`
- Input shape: `(batch, 6, 750)`
- Model: IFNet

### Common8

Common8 is not included in this release because no verified Cho2017 source-only Common8 checkpoint was found. HS patient-trained downstream checkpoints are not substitutes for a source-only pretraining checkpoint.

### Legacy full22

The previous 22-channel IFNet and EEGNeX checkpoints remain legacy lineages. Full22 checkpoints must not be loaded with Common10 or Common6 inputs.

## HS Stroke downstream evaluation

The Cho2017-pretrained IFNet representation was evaluated on a longitudinal stroke MI-EEG cohort of 14 independent patients using patient-held-out OOF evaluation. Repeated sessions from the same patient were never split across folds.

### Classification results

- Independent patients: 14
- Pretrained IFNet mean accuracy: 50.48%
- Pretrained IFNet mean balanced accuracy: 50.48%
- Pretrained IFNet mean macro F1: 47.55%
- Pretrained IFNet mean class 0 recall: NA
- Pretrained IFNet mean class 1 recall: NA
- Random-frozen IFNet mean balanced accuracy: 50.03%
- Pretrained minus random-frozen balanced accuracy: 0.45%p
- Patient-level wins/ties/losses: 9/1/4

Metrics shown as `NA` were not consistently recoverable from the locked per-fold JSON outputs and are therefore not reported.

### Representations prepared for clinical regression

Three aligned representation groups were prepared for internal collaboration:

1. Cho2017-pretrained IFNet OOF learned features
2. Random-frozen IFNet control features
3. Interpretable physiological EEG features

Each IFNet assessment representation contains 1,024 predictors: the 512D mean trial representation and a 512D class-1-minus-class-0 contrast.

- Independent patients: 14
- Assessment-level rows: 28
- Patient-level change rows: 14
- Pretrained/random trial alignment: passed
- Pretrained/random assessment alignment: passed

This is a small-sample feasibility analysis. Although the assessment table contains 28 rows, the independent sample size is 14 patients. Evaluation must use patient-level LOSO. Scaling, PCA, feature selection, and hyperparameter selection must be fitted within each training fold.

## Public-release scope

Patient EEG, FMA scores, patient-level OOF features and predictions, and HS downstream checkpoints are not released. Only aggregate results and protocol descriptions are included.

## Integrity

Checkpoint SHA-256 hashes and parameter shapes are provided in `manifests/channel_lineage_manifest.json`.


<!-- Legacy README was archived in reports/README_legacy_full22_before_channel_update.md -->
