# DQS preference post-training runtime

이 디렉터리는 연구·데이터 합성 작업공간이 아니라 외부 GPU 인스턴스에서
`mPO`, `CPO`, `DPO`를 실행하기 위한 생성형 배포 패키지다. SFT 모델 자체와
평가 런타임은 Git에 포함하지 않으며, SFT 모델은 고정된 Hub revision에서 먼저
명시적으로 내려받는다.

## 포함 범위

- mPO: Teacher post-edit 전체 SFT + term-token masked preference loss
- CPO: Teacher 전체 응답 대 Student 전체 응답
- DPO: 같은 full-response pair와 고정된 초기 SFT reference log-probability
- objective별 config, strict dataset contract, smoke/full training entry point
- 파일 SHA256과 SFT provenance 요구사항을 담은 `manifest.json`

raw golden pairs, 합성·필터링 코드, source review, 분석 보고서, 기존 출력과
캐시는 포함하지 않는다. 학습 후 평가는 원래 DQS 평가 파이프라인에서 별도
output 경로로 수행한다.

## 실행 순서

CUDA/PyTorch는 인스턴스에 맞게 먼저 설치한다. 그 뒤 나머지 고정 버전을
설치하고 패키지를 검증한 다음, 학습 전에 세 데이터와 SFT final 모델을
명시적으로 내려받는다.

```bash
make set
make validate
make download-data
make download-model
make validate-data
make validate-model
make validate-runtime
make test
make dry-run
```

`make download-data`는 mPO/CPO/DPO JSONL과 원격 contract를 공개 dataset의 exact
40-hex commit에서 임시 staging 디렉터리로 모두 내려받는다. 세 objective의 원격
contract SHA256, JSONL SHA256, JSONL 문법, 행 수가 전부 맞아야만
`data/train/{mpo,cpo,dpo}.jsonl` 설치 단계로 넘어간다. 검증에 실패한 파일은
설치하지 않는다. 이미 정확한 파일은 재사용하고, 기존 파일이
손상된 경우에는 자동 교체하지 않는다.

```bash
make download-data DOWNLOAD_WORKERS=16
# 손상 파일을 검증된 원격 파일로 명시적으로 교체할 때만 사용
make download-data DOWNLOAD_REPLACE=1
```

trainer는 `data.source: local`만 허용한다. 따라서 `make download-data` 이후의
dry-run/smoke/full train에서는 네트워크 다운로드가 일어나지 않으며 데이터가
없으면 즉시 실패한다.

`make download-model`은 `alwaysgood/dqs-runs`의 exact 40-hex commit에서 해당
run의 `checkpoints/final/` 8개 파일(총 10,279,726,920 bytes)만 임시 staging으로
받는다. 모든 파일의 크기와 SHA256, 전체 파일 목록, `dqs_stage_model.json`의
run/subset/global-step/full-tuning 값이 전부 맞아야 `models/sft_final`에 설치한다.
`checkpoint-184/`의 optimizer state는 다운로드하지 않으며 다른 모델이나 최신
branch로 fallback하지 않는다.

기존 모델이 정확하면 네트워크를 사용하지 않고 재사용한다. 기존 파일이
손상됐으면 자동 교체하지 않으며, 알려진 8개 파일 외 다른 파일이 없는 경우에만
명시적으로 교체할 수 있다.

```bash
make download-model MODEL_DOWNLOAD_WORKERS=16
make download-model MODEL_REPLACE=1
```

SFT 최종 모델은 기본적으로 `models/sft_final`에 둔다. 다른 위치를 쓰면 다운로드와
학습의 모든 명령에 동일한 `MODEL_DIR`를 넘긴다.

```bash
make validate-model MODEL_DIR=/workspace/models/sft_final
```

각 objective는 반드시 동일 환경에서 one-step smoke를 먼저 성공시켜야 한다.
smoke receipt가 현재 코드·데이터·모델·런타임 hash와 다르면 full run은 시작하지
않는다.

```bash
make smoke-mpo MODEL_DIR=/workspace/models/sft_final
make train-mpo MODEL_DIR=/workspace/models/sft_final

make smoke-cpo MODEL_DIR=/workspace/models/sft_final
make train-cpo MODEL_DIR=/workspace/models/sft_final

make smoke-dpo MODEL_DIR=/workspace/models/sft_final
make train-dpo MODEL_DIR=/workspace/models/sft_final
```

멀티 GPU는 smoke와 full 모두 같은 `LAUNCH`를 넘긴다.

```bash
make smoke-mpo \
  MODEL_DIR=/workspace/models/sft_final \
  LAUNCH='torchrun --nproc_per_node=2'
make train-mpo \
  MODEL_DIR=/workspace/models/sft_final \
  LAUNCH='torchrun --nproc_per_node=2'
```

## 데이터와 캐시

배포 직후 `data/train/`은 비어 있다. `manifest.json`의 `data_mode: hf`는 데이터의
배포 원천이 Hub라는 뜻이고, `data_access: explicit_download_then_local`이 실제 접근
방식을 고정한다. 세 config는 항상 `data/train/*.jsonl`만 읽는다. branch나 `main`,
자동 최신 revision은 허용하지 않는다.

`.cache/datasets/`는 원본 데이터 보관소가 아니라, 로컬 JSONL을
`datasets.load_dataset()`이 학습용 Arrow 형식으로 읽으며 만드는 런타임 캐시다.
Git/배포 artifact에는 포함되지 않고 학습이 모두 끝난 뒤 삭제할 수 있다. 기존
`hf_cache/` 경로는 사용하지 않는다.

## 변경하면 새 smoke가 필요한 항목

- 코드 또는 config
- 데이터/contract
- SFT 모델 파일
- CUDA GPU와 bf16 실행 환경
- 고정된 Python package 버전

학습 중 다른 backend, full-logits 경로, truncation, reference-free DPO, W&B
fallback으로 전환하는 경로는 없다. 계약이 맞지 않으면 즉시 실패한다.
