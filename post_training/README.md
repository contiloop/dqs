# DQS preference post-training

`post_training/`은 배포 학습 코드와 데이터 합성 연구 코드를 분리한다. 외부 GPU
인스턴스에서 사용하는 canonical 패키지는
`dqs_preference_training_hf/` 하나다.

## 디렉터리

```text
post_training/
├── dqs_preference_training_hf/  # 외부 인스턴스용 mPO/CPO/DPO 패키지
│   ├── configs/
│   ├── src/
│   ├── data/contracts/
│   ├── scripts/
│   └── tests/
├── research/                    # 합성, 필터링, 감사 자료와 재현 코드
│   ├── data_prep/
│   ├── audits/
│   ├── evaluation/
│   ├── configs/
│   ├── contracts/
│   └── tests/
├── scripts/build_release.py     # 독립 배포 bundle/archive 생성기
└── tests/                       # runtime 및 release 회귀 테스트
```

배포 패키지에는 raw golden pairs, source-quality review, 합성 중간 산출물 또는
평가 runtime이 들어가지 않는다. 공개 HF dataset을 exact commit으로 명시적으로
다운로드한 뒤 로컬 JSONL만 학습한다.

## 외부 학습

새 인스턴스에서 clone부터 학습까지 이어지는 전체 명령은
`dqs_preference_training_hf/README.md`의 `Post-training quick run`을 따른다.

```bash
cd post_training/dqs_preference_training_hf

make set
make validate
make download-data DOWNLOAD_WORKERS=16
make download-model MODEL_DOWNLOAD_WORKERS=16
make validate-data
make validate-runtime
make validate-model MODEL_DIR=/workspace/models/sft_final
make test
make dry-run MODEL_DIR=/workspace/models/sft_final

make smoke-mpo MODEL_DIR=/workspace/models/sft_final
make train-mpo MODEL_DIR=/workspace/models/sft_final
```

CPO와 DPO는 `smoke-cpo/train-cpo`, `smoke-dpo/train-dpo`로 실행한다. 세 objective는
서로의 결과를 이어받지 않고 동일한 SFT final checkpoint에서 독립적으로 시작한다.

## 유지보수 검증

```bash
make -C post_training test
make -C post_training build-release
```

데이터 합성의 세부 근거와 재현 순서는 `research/DATA_PREP_PLAN.md`, loss와 runtime
구현 기록은 `research/IMPLEMENTATION_NOTES.md`에 보존한다.
