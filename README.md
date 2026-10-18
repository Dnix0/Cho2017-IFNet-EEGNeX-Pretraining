# Cho2017 기반 IFNet·EEGNeX 사전학습 모델 인계 패키지

## 1. 개요

이 저장소는 Cho2017 운동상상 EEG 데이터로 사전학습한 **IFNet**과 **EEGNeX** 모델을 환자 데이터 본학습 담당자에게 인계하기 위한 패키지입니다.

이 저장소가 제공하는 범위는 **본학습 직전까지의 사전학습 모델, 검증 결과, 로딩 코드 및 고정 앙상블 규칙**입니다. 환자 데이터의 구성, 환자 단위 분할, fine-tuning, 내부·외부 평가 및 최종 성능 분석은 별도의 downstream 단계에서 수행해야 합니다.

> 원본 Cho2017 데이터와 환자 데이터는 이 저장소에 포함하지 않습니다.

---

## 2. 제공 모델

### IFNet

* 사전학습 데이터: Cho2017 전체 52명
* 입력 형태: `(batch, 22, 750)`
* Sampling rate: 250 Hz
* 입력 구간: 0–3초
* 사전학습 과제: 왼손 대 오른손 운동상상 이진분류
* 학습 길이: 100 epochs
* 선택된 best epoch: 75
* 내부 source-validation accuracy: 81.57%
* 체크포인트:
`checkpoints/ifnet_cho2017_fullsource_full_seed42_100ep_best.pth`
* SHA-256:
`484cb27c04adeef9284d95365c4610e9671aafa96f87aa8bd4d79864c3140202`

### EEGNeX

* 사전학습 데이터: Cho2017 전체 52명
* 입력 형태: `(batch, 22, 750)`
* Sampling rate: 250 Hz
* 입력 구간: 0–3초
* 사전학습 과제: 왼손 대 오른손 운동상상 이진분류
* 학습 길이: 50 epochs
* 선택된 best epoch: 49
* 내부 source-validation accuracy: 82.89%
* 체크포인트:
`checkpoints/eegnex_cho2017_fullsource_full_seed42_50ep_best.pth`
* SHA-256:
`2d9c14ad4a4cfdf40e7f05f8e0d7710d05ebfb55bfc338a34ad8086d6d5d05d0`

내부 source-validation 성능은 full-source checkpoint 선택을 위한 값입니다. 새로운 피험자 또는 환자 데이터의 성능으로 해석하면 안 됩니다.

---

## 3. 검증 결과 요약

Cho2017 52-subject LOSO, seed 42 기준 주 결과는 다음과 같습니다.

|방법|Subject-mean Accuracy|Mean Macro-F1|Pooled Accuracy|
|-|-:|-:|-:|
|IFNet OFF|77.49%|76.73%|78.13%|
|EEGNeX OFF|78.46%|77.36%|78.92%|
|**IFNet 0.75 + EEGNeX 0.25 Ensemble OFF**|**79.27%**|**78.52%**|**79.76%**|
|Target-BN|74.64%|74.10%|75.08%|
|Target-BN → T3A|70.44%|69.69%|70.56%|
|Dual-Path TTA|73.62%|72.90%|74.03%|

상세 결과는 `validation/`과 `reports/`에서 확인할 수 있습니다.

---

## 4. 권장 downstream 사용 방법

두 모델은 환자 데이터에서 **각각 독립적으로 fine-tuning**해야 합니다.

1. IFNet 체크포인트를 로드합니다.
2. 환자 데이터에 맞게 IFNet을 fine-tuning합니다.
3. EEGNeX 체크포인트를 별도로 로드합니다.
4. 동일한 환자 데이터 분할에서 EEGNeX를 fine-tuning합니다.
5. 두 모델의 softmax probability를 고정 비율로 결합합니다.

```python
import torch

p_ifnet = torch.softmax(ifnet_logits, dim=1)
p_eegnex = torch.softmax(eegnex_logits, dim=1)

p_final = 0.75 \* p_ifnet + 0.25 \* p_eegnex
pred = p_final.argmax(dim=1)
```

이 결합은 **logit 평균이 아니라 probability-level 앙상블**입니다.

`0.75:0.25`는 사전 지정된 주 평가 후보입니다. 환자 데이터에서도 최적의 가중치라고 보장하는 것은 아니며, protected test 결과를 본 뒤 가중치를 다시 선택해서는 안 됩니다.

---

## 5. 체크포인트 로드 예시

```python
from src.load_pretrained_for_finetuning import load_cho2017_pretrained

ifnet, ifnet_meta, _ = load_cho2017_pretrained(
    model_name="ifnet",
    checkpoint_path=(
        "checkpoints/"
        "ifnet_cho2017_fullsource_full_seed42_100ep_best.pth"
    ),
    device="cuda",
)

eegnex, eegnex_meta, _ = load_cho2017_pretrained(
    model_name="eegnex",
    checkpoint_path=(
        "checkpoints/"
        "eegnex_cho2017_fullsource_full_seed42_50ep_best.pth"
    ),
    device="cuda",
)
```

환자 데이터의 class 수 또는 label 의미가 Cho2017과 다르면 기존 2-class classifier head를 교체하고 backbone 가중치를 초기값으로 사용해야 합니다.

---

## 6. 본학습 전에 반드시 확인할 사항

* 환자 데이터의 EEG 채널 수와 채널 이름
* 22개 채널의 순서가 사전학습 조건과 일치하는지
* Sampling rate가 250 Hz인지
* 입력 구간이 3초인지
* 입력 tensor가 `(batch, channels, samples)` 순서인지
* 환자 데이터의 class 수와 label 정의
* 동일 환자의 trial이 train, validation, test에 중복되지 않는지
* 정규화와 alignment 통계가 training subset에서만 계산되는지
* classifier head를 유지할지 재초기화할지

입력 조건이 다르면 무조건 체크포인트를 바로 적용하지 말고, 채널 매핑·resampling·crop 또는 padding 규칙을 먼저 고정해야 합니다.

---

## 7. TTA 사용 여부

**기본 설정은 TTA OFF입니다.**

다음 방법은 Cho2017 52-subject LOSO에서 구현 무결성 검사를 통과했지만 평균 성능을 낮췄습니다.

* Target-BN
* All-Support T3A
* Dual-Path TTA

따라서 환자 데이터 본학습이나 평가에서 자동으로 활성화하지 마세요. 향후 TTA를 다시 평가하려면 target label을 adaptation이나 방법 선택에 사용하지 않는 별도의 사전 정의 프로토콜이 필요합니다.

---

## 8. 전처리 계보

Full-source 사전학습에는 다음 구성을 사용했습니다.

* Cho2017 전체 52명
* Train: 8,763 trials
* Internal validation: 982 trials
* 각 피험자의 training trial만으로 subject-wise RA reference 계산
* Validation trial은 RA reference 계산에 사용하지 않음
* 전체 training pool에서 global scalar mean/std 계산
* Target-dependent dispersion scaling 사용하지 않음

기록된 training 통계:

* Global train mean: `-0.014183156192302704`
* Global train std: `1.4661153554916382`

세부 설정은 `configs/ifnet/`과 `configs/eegnex/`에 있습니다.

---

## 9. 저장소 구조

```text
.
├─ checkpoints/
│  ├─ ifnet_cho2017_fullsource_full_seed42_100ep_best.pth
│  └─ eegnex_cho2017_fullsource_full_seed42_50ep_best.pth
├─ configs/
│  ├─ ifnet/
│  └─ eegnex/
├─ manifests/
│  ├─ file_manifest.csv
│  └─ handoff_summary.json
├─ reports/
│  └─ Cho2017_IFNet_EEGNeX_Pretraining_Final_Handoff_Report_20261018.docx
├─ src/
│  ├─ ensemble_probability.py
│  ├─ load_pretrained_for_finetuning.py
│  └─ models/
├─ validation/
│  ├─ evaluation_config.json
│  ├─ seed42_subject_metrics.csv
│  ├─ seed42_summary.csv
│  └─ seed42_tta_audit.csv
└─ README.md
```

---

## 10. 파일 무결성 확인

다운로드 후 체크포인트 SHA-256을 확인하는 것을 권장합니다.

### Windows

```bat
certutil -hashfile "checkpoints\\ifnet_cho2017_fullsource_full_seed42_100ep_best.pth" SHA256
certutil -hashfile "checkpoints\\eegnex_cho2017_fullsource_full_seed42_50ep_best.pth" SHA256
```

### Python

```python
from pathlib import Path
import hashlib

for path in sorted(Path("checkpoints").glob("\*.pth")):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(path.name, digest)
```

전체 파일은 `manifests/file_manifest.csv`에 기록된 SHA-256과 비교할 수 있습니다.

---

## 11. 데이터 및 재현 범위

* 원본 Cho2017 데이터는 저장소에 포함하지 않습니다.
* 환자 데이터는 저장소에 포함하지 않습니다.
* 저장소의 LOSO 결과는 Cho2017 seed 42 결과입니다.
* Cho2017 결과를 환자 데이터 성능으로 해석해서는 안 됩니다.
* 환자 데이터 본학습 결과는 별도의 환자 단위 validation/test로 검증해야 합니다.

---

## 12. 라이선스 및 인용

이 저장소의 코드와 가중치는 저장소에 포함된 라이선스 조건을 따릅니다. IFNet, EEGNeX 및 Cho2017을 연구 결과에 사용할 경우 원 논문과 데이터셋 출처를 확인하여 적절히 인용하세요.

원본 데이터는 재배포하지 않으며, 사용자는 Cho2017의 공식 배포 경로와 이용 조건을 따라 직접 확보해야 합니다.

