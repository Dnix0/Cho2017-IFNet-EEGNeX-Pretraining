# GitHub 업로드 전 체크리스트

- Common10 IFNet·EEGNeX와 Common6 IFNet만 포함됐는지 확인
- Common8이 미포함 상태로 명시됐는지 확인
- `new_dataset_audit\liu2024...`의 환자별 checkpoint가 없는지 확인
- HS downstream/fine-tuned checkpoint가 없는지 확인
- 환자 EEG, FMA, OOF feature가 없는지 확인
- manifest의 SHA-256과 파일 크기 확인
- GitHub 웹 업로드 시 기존 README와 checkpoint 폴더를 staging 내용으로 교체
