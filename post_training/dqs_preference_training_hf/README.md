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

## Post-training quick run

아래는 새 외부 GPU 인스턴스에서 시작하는 전체 명령이다. 먼저 CUDA와 호환되는
PyTorch가 설치된 이미지가 필요하며, 이 패키지는 `2.4 <= torch < 2.11`, CUDA,
bf16 지원 GPU를 요구한다. `mPO`, `CPO`, `DPO`는 서로 이어서 학습하는 단계가
아니라 동일한 SFT final 모델에서 각각 시작하는 독립 실험이다.

```bash
# 1. 코드 받기
cd /workspace
git clone --branch codex/post-training --single-branch \
  https://github.com/contiloop/dqs.git
cd dqs/post_training/dqs_preference_training_hf

# 2. GPU/PyTorch가 들어 있는 인스턴스 환경을 보존한 venv 만들기
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make set

# 3. CUDA와 bf16 확인
nvidia-smi
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("this post-training package requires bf16 support")
PY

# 4. strict online W&B 인증
wandb login

# 5. 다운로드 전에 코드·환경 테스트
make validate
make validate-runtime
make test

# 6. 공개 HF에서 preference 데이터 3종과 SFT final 모델 1개 다운로드
df -h .
make download-all DOWNLOAD_WORKERS=16 MODEL_DOWNLOAD_WORKERS=16

# 7. 다운로드 결과와 세 objective config 검증
make validate-data
make validate-model
make dry-run
```

이후 실제로 실행할 objective 하나를 고른다. 각 full run은 대응하는 one-step
smoke receipt가 반드시 먼저 있어야 한다.

```bash
# mPO
make smoke-mpo
make train-mpo

# CPO를 실행할 때는 위 mPO 두 명령 대신 아래 두 명령 사용
# make smoke-cpo
# make train-cpo

# DPO를 실행할 때는 위 mPO 두 명령 대신 아래 두 명령 사용
# make smoke-dpo
# make train-dpo
```

Cosine run과 동일한 SFT 초기화 및 loss weight를 유지하면서, warm-up과
gradient clipping 없이 `5e-6`을 끝까지 사용하는 constant-LR 실험은 별도
config로 실행한다. 이 config의 `max_grad_norm: 0.0`은 pre-clip norm 로깅은
유지하지만 optimizer에 전달되는 gradient를 축소하지 않는다.

```bash
make train-mpo MPO_CONFIG=configs/mpo_constant.yaml
```

이 config는 기존 cosine run과 output directory 및 W&B run ID가 다르므로
기존 결과를 resume하거나 덮어쓰지 않는다. 기존 환경에서 이미 Unsloth one-step
검증을 완료했으므로 이 config는 smoke receipt를 요구하지 않는다. 필요할 때만
`make smoke-mpo MPO_CONFIG=configs/mpo_constant.yaml`을 별도로 실행할 수 있다.

여러 GPU를 사용할 때는 smoke와 full train에 완전히 동일한 `LAUNCH`를 넘긴다.
아래 예시는 mPO 4-GPU 실행이다.

```bash
make smoke-mpo LAUNCH='torchrun --standalone --nproc_per_node=4'
make train-mpo  LAUNCH='torchrun --standalone --nproc_per_node=4'
```

기본 결과 경로는 각각 다음과 같다.

- mPO: `outputs/gemma4_e2b_dqs_mpo_setting5_seed42/final`
- CPO: `outputs/gemma4_e2b_dqs_cpo_full_response_seed42/final`
- DPO: `outputs/gemma4_e2b_dqs_dpo_full_response_seed42/final`

HF 데이터와 모델 저장소는 공개이므로 HF 로그인은 필수가 아니다. 입력 SFT
모델만 10,279,726,920 bytes이고 full-training checkpoint에는 optimizer state도
저장되므로, 실행 전에 `df -h`로 충분한 디스크 공간을 확인한다.

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

`make set`은 의도적으로 두 단계다. 먼저 `requirements-gpu.txt`로
`unsloth==2026.7.2`의 공개 PyPI dependency metadata와 호환되는 환경을 만든 뒤,
`requirements-transformers-gemma4.txt`를 `--no-deps`로 적용해 최종 런타임을
`transformers==5.5.3`으로 고정한다. 이 버전은 Gemma-4 E-series의 gradient
checkpointing에 필요한 function-scoped shared-KV 경로와 checkpoint-safe keyword
전달을 제공한다. mPO/CPO는 chosen/rejected를 배치 축으로 이어 붙인 `[2B, L]`
tensor를 정확히 한 번 forward하고, 양쪽에서 필요한 position의 합집합만 projection한
뒤 각각의 독립 mask로 loss를 정규화한다.
`5.5.0`부터 `5.5.2`까지는 허용하지 않으며,
`make validate-runtime`이 최종 버전을 fail-closed로 검사한다.

Gemma-4 E2B의 cross-layer shared-KV는 정확한 gradient를 위해 non-reentrant
activation checkpointing을 사용한다. pinned Torch/Unsloth 조합에서 compiled
decoder-layer variant가 backward 재계산 중 다른 순서로 선택되면 forward와 recompute의
tensor metadata가 달라질 수 있다. 따라서 이 release는
`training.unsloth_compile: disabled`와 `UNSLOTH_COMPILE_DISABLE=1`을 필수 계약으로
둔다. 이는 Transformers backend로의 fallback이 아니다. Unsloth `FastModel`, Gemma4
patch, fused large-vocab CE, full BF16 training과 gradient checkpointing은 그대로
사용하고 `torch.compile`만 끈다. entry point는 Unsloth가 먼저 import된 경우에도
조용히 진행하지 않고 재시작을 요구한다.

`warmup_steps: 0.1`의 float 값은 전체 optimizer step의 10%라는 뜻이다. 이전
`warmup_ratio: 0.1`과 동일한 schedule이며 Transformers v5 deprecation warning만
제거한다.

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

이미 이전 release의 데이터를 `data/train/`에 받은 상태에서 Git branch를 새
dataset revision으로 갱신했다면, 기존 파일은 새 contract와 의도적으로 맞지
않는다. 이 경우에만 `make download-data DOWNLOAD_REPLACE=1`로 세 파일을 새 exact
revision의 검증된 파일로 교체한다.

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

세 데이터 contract의 completion EOS는 최종 SFT tokenizer와 같은 `<turn|>`
(token id 106)이다. `validate-model`은 다운로드한 tokenizer/model/generation
config와 이 EOS profile을 함께 검사하며, base tokenizer의 EOS id 1로 자동
변환하거나 우회하지 않는다.
