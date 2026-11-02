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



---

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

