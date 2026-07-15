# DQS terminology mPO post-training

## 외부 GPU 배포 패키지

최종 JSONL 세 종류는 공개 Hugging Face dataset을 canonical source로 사용한다.
로컬 `prepared/`, `raw/`, `source_quality/`, `.cache/`는 재생성 가능한 산출물이므로
보존하거나 배포 패키지에 복사하지 않는다. 아래 생성기는 SFT 단계와 같은
`Makefile + configs + src + scripts + tests` 형태의 code-only 패키지만 만든다.

```bash
python3 post_training/scripts/build_release.py \
  --output post_training/dist/dqs_preference_training \
  --data-mode hf \
  --hf-repo-id alwaysgood/dqs-post-training \
  --hf-revision 0f7b051f96b3ccdc3837f9537e5aac3a776bf4f1 \
  --replace \
  --archive
```

HF branch 이름이나 `main`은 허용하지 않는다. 생성된 패키지의 세부 실행 순서는
`post_training/dist/dqs_preference_training/README.md`에 있다.

배포 패키지에서 `make download-data`를 먼저 실행하면 exact commit의 세 JSONL을
`data/train/`에 전부 내려받아 SHA256/contract/행 수를 검증한다. trainer는 이후
로컬 파일만 읽으며 학습 중 자동 HF 다운로드는 허용하지 않는다.

공개 dataset: <https://huggingface.co/datasets/alwaysgood/dqs-post-training>

## 무엇을 구현했는가

현재 trainer는 논문의 **setting 5**만 구현한다.

```text
L = lambda_sft * L_SFT(y+) + lambda_mpo * L_mPO(y+, y-)
```

- `L_SFT`: `y+`의 prompt를 제외한 전체 completion 토큰 평균 NLL
- `L_mPO`: `M+`와 `M-`에서 각각 독립적으로 평균한 term log-probability의 margin SmoothL1
- `margin = mean_logp(y+, M+) - mean_logp(y-, M-)`
- `target_margin = 1 / (2 * preference_beta)`
- 논문값 `preference_beta=0.25`이면 target margin은 `2.0`
- 논문 preset은 `lambda_sft=10`, `lambda_mpo=1`

논문의 setting 6은 `SFT + mSFT + PO + mPO` 네 항이다. 지금 데이터 설계에서 요청한 두 항 결합은 setting 5이므로 코드가 `paper_setting: 6`을 조용히 받아들이지 않고 즉시 거부한다.

논문 식 (4)/(6)은 atomic SFT 식 안에 `alpha`를 적었고 식 (7)은 SFT block 바깥에 다시 `alpha`를 적어 표기가 중복되어 있다. 이 구현에서는 atomic loss를 무가중으로 정의하고 최종 합산에서 coefficient를 **정확히 한 번만** 적용한다.

## 기존 SFT와 겹치지 않는 경계

이 코드는 기존 `src/sft_train.py`를 호출하거나 수정하지 않는다.

1. `model.name_or_path`에서 기존 full-SFT의 `checkpoints/final` **가중치만** 읽는다.
2. post-training의 optimizer, scheduler, global step은 새로 만든다.
3. 기존 `checkpoint-184`의 optimizer state를 이어받지 않는다.
4. post-training 재개는 자기 `post_training/outputs/.../checkpoint-*`만 허용한다.
5. 모든 출력은 `post_training/outputs/` 아래에 저장한다.

기본 optimizer hyperparameter는 이 모델을 만든 실제 iterative full-SFT snapshot을
기준으로 한다. float `warmup_steps: 0.1`(전체 step의 10%), cosine scheduler, AdamW, max gradient norm
`5.0`, weight decay `0.0`, effective batch size `128`, 1 epoch은 유지하되, localized
full-model post-training의 peak learning rate는 SFT의 `2e-5`보다 4배 낮은 `5e-6`을
쓴다. 이는 optimizer state를 재개한다는 뜻이 아니다. post-training은 새 optimizer와
새 scheduler를 시작한다. mPO는 chosen/rejected를 `[2B, L]` 단일 결합 forward로
처리하므로 pair 기준 per-device micro-batch는 SFT의 `4` 대신 `1`을 쓰고 gradient
accumulation으로 같은 global batch를 만든다.

따라서 post-training loss 안의 `L_SFT(y+)`는 “기존 SFT를 한 번 더 실행”한다는 뜻이 아니다. 같은 preference batch에서 전체 번역 품질을 붙잡는 regularizer이며, mPO와 **동시에 한 optimizer step**으로 계산된다.

모델 로딩은 LoRA/PEFT가 아닌 `full_finetuning=True`다. 기본값
`training.freeze_embeddings: false`에서는 text embedding을 포함한 사용 가능한
text-model weight를 학습한다. Gemma4의 vision/audio 모듈과 forward에서 구조적으로
사용되지 않는 shared-KV 파라미터만 제외한다.

`training.freeze_embeddings: true`로 바꾸면 input token embedding만 추가로
고정한다. input embedding과 LM head가 tied weight이면 같은 파라미터이므로 LM
head의 공유 weight도 함께 고정한다. 이를 조용히 untie하지 않으며 실제 tied 여부와
freeze 결과를 run manifest에 기록한다. 현재 `google/gemma-4-E2B-it` config는
`tie_word_embeddings=true`이므로 이 옵션을 켜면 LM head의 공유 weight도 실제로
고정된다.

## mask와 계산 계약

- prompt token 제외
- right-padding token 제외
- `logits[:, t-1] -> input_ids[:, t]` causal shift 반영
- chosen/rejected term mask 독립 사용
- 각 행에서 completion, `M+`, `M-`를 각자의 token 수로 먼저 정규화
- 여러 term span의 모든 token을 합집합으로 포함
- shift 후 빈 mask면 즉시 실패
- Gemma4 text-only 학습에는 `mm_token_type_ids=0`을 명시적으로 전달
- chosen selected logits에는 fused CE autograd node를 한 번만 만들고, 그 per-token
  log-probability를 SFT completion mask와 mPO term mask가 공유

Gemma4의 큰 vocabulary 때문에 전체 sequence logits를 만드는 경로는 운영 trainer에서 허용하지 않는다. chosen forward는 SFT에 필요한 completion prediction position만, rejected forward는 mPO에 필요한 term prediction position만 tensor `logits_to_keep`로 projection한다. 모델이 이 tensor를 무시하면 shape contract가 실패하며 학습은 중단된다.

마지막 항목은 단순 최적화가 아니다. Unsloth fused CE는 backward에서 저장된
logits 버퍼를 gradient 버퍼로 재사용하므로, 같은 chosen logits에 CE node를 두 개
만들지 않는다. 하나의 token-logp tensor에서 두 mask를 독립 정규화해 합산한다.

## 최종 source-filtered 학습 데이터

strict-v2는 기존 5,888행을 다시 거르는 산출물이 아니다. 23개 raw
`golden_pairs` 파일을 처음부터 한 번 읽어, 각 terminology row를 즉시 완전한 pair로
합성하거나 rejection ledger에 기록한다. 그 5,505건의 source를 전수 판정하고,
93개 REVIEW도 직접 재검토해 source-quality KEEP 5,201건을 확정했다. 이 중 raw
Teacher/Student 응답은 동일하지만 terminology annotation만 남은 불일치 1건을
preference 단계에서 fail-closed하여, 현재 canonical mPO 데이터는 5,200건이다.

```text
data.source:        local
data.path:          data/train/mpo.jsonl
data.contract_path: data/contracts/mpo.json
data.cache_dir:     .cache/datasets/mpo
```

| 단계 | rows |
|---|---:|
| raw golden rows | 23,000 |
| terminology rows | 9,052 |
| base exact-mapping reject | 1,217 |
| strict synthesis reject | 1,624 |
| strict char-span pair | 6,211 |
| tokenizer boundary reject | 706 |
| strict-v2 token pair / source review 입력 | 5,505 |
| source-quality reject | 304 |
| source-quality KEEP | 5,201 |
| preference annotation 불일치 reject | 1 |
| 최종 mPO train pair | **5,200** |
| Teacher-vs-Student full-response DPO/CPO pair | **5,200** |

`build_preference_pairs_strict_v2.py`는 prior candidate/tokenized artifact를 읽지 않는다.
한 annotation이라도 실패하면 행 전체를 버리며 partial recovery, 문자열 repair,
fallback은 없다. 기존 source exact-match/조사/구두점/괄호/길이 경고를 모두 hard
reject하고, Teacher 문맥과 student term을 접합하며 새로 생긴 경계 중복, 조사·어미
충돌, 괄호·인용부호 구조 손상, 문장 전체형 annotation, completion 전체를 덮는 term
mask도 제외한다. 서로 다른 두 annotated term이 인접한 경우와 student 원출력에 이미
존재한 반복은 새 splice defect로 오인하지 않는다.

최종 데이터에는 quality warning, 긴 target term, completion 전체 term mask가 0개다.
띄어쓰기·하이픈·퍼센트 단위처럼 Teacher가 실제 terminology correction으로 표시한
13행은 token contrast가 존재하므로 유지한다. 기존 중간/v1 데이터 파일은 로컬에서
삭제하고, 비교에 필요한 고정 contract와 최종 검증 보고서만 보존한다.

source review는 저장된 5,505개 판단을 결합한 뒤 `KEEP 5,196 / REJECT 216 /
REVIEW 93`을 얻었다. REVIEW 원문을 직접 읽어 명백한 heading/document boundary 유실
5건만 KEEP으로 승격하고 88건을 제외했다. 최종 결과는 `KEEP 5,201 / REJECT 304`이며
REVIEW는 남지 않는다. 표 layout 손실, 절단, 혼합 문맥은 치환이나 복원 없이 제외한다.

source-quality KEEP 중 annotation 불일치 1건은 mPO와 full-response preference에서
모두 제외한다. 남은 동일한 5,200건으로 DPO/CPO용 데이터도 별도로 만든다. 여기서는 mPO의 합성
minimal negative를 쓰지 않고 `chosen=raw target(Teacher post-edit)`,
`rejected=raw student_translation(Student 전체 출력)`을 사용한다. prompt는 이미 chat template이
직렬화되어 있으므로 다시 template을 적용하면 안 된다.

full-response baseline은 **DPO와 CPO 모두 실행 가능**하다. 둘 다 같은 Teacher-vs-Student
5,200 pair를 쓰지만 학습 입력과 목적함수는 분리한다. CPO는 공용 raw pair를 동일 tokenizer로
다시 전수 정렬해 Hub의 `cpo/train.jsonl`로 고정한 데이터를 사용한다. Teacher와 Student
completion 전체(EOS 포함)를 각각 독립 mask로 두며, 최대 sequence는 2,825 tokens라
`max_seq_length=2,908`에서 truncation이 없다. CPO preference 항은 completion log-prob
합의 차이를 쓰고, Teacher completion NLL은 token 평균으로 별도 정규화한다.

```text
L_CPO = -logsigmoid(beta * (sum logp(Teacher) - sum logp(Student)))
        + cpo_alpha * NLL(Teacher)
```

Gemma4 text-only selected-logit forward를 직접 쓰므로 prompt logits/full-logits fallback은
없다. DPO는 Hub의 `dpo/train.jsonl`에 저장된 표준
`prompt/chosen/rejected`를 `UnslothDPOTrainer`에 직접 전달한다.

```text
L_DPO = -logsigmoid(beta * ((log pi(Teacher) - log pi(Student))
                            - (log ref(Teacher) - log ref(Student))))
```

DPO에는 SFT/RPO 항을 섞지 않는다. `ref_model` 전체 복제도 만들지 않고, DPO update 또는
resume checkpoint restore 전에 원래 SFT policy의 completion log-prob 합을 5,200행에
선계산해 고정한다. `trl==0.24.0`의 DPO class가 실제로 Unsloth의
`UnslothDPOTrainer`/`UnslothDPOConfig`로 패치됐는지 확인하며, full Gemma4 Processor,
text-only all-zero `mm_token_type_ids`, completion EOS, no-truncation, `logits_to_keep` 실제
반영을 모두 hard assertion한다. pristine TRL trainer나 bare tokenizer로 돌아가는 경로는 없다.

SFT final 저장본의 tokenizer는 completion EOS가 `<turn|>`(id 106)이다. 최초 데이터
token mask는 동일한 vocabulary/backend의 base tokenizer EOS id 1로 만들어졌으므로,
release 직전에 `post_training/scripts/retarget_preference_eos.py`로 mPO/CPO 양쪽의 마지막
appended EOS만 106으로 바꿨다. 5,200행의 본문 token, sequence length, completion/term mask는
전부 그대로이며 DPO raw text는 byte-for-byte 그대로다. 세 contract는 exact SFT final
tokenizer config SHA256과 EOS 106을 묶고, 런타임에서 id 1로 되돌리는 fallback은 없다.

아래는 원천 데이터부터 전부 다시 합성할 때만 쓰는 재생성 절차다. 삭제된 raw source와
tokenizer cache를 먼저 복원해야 하며, 생성되는 `prepared/`와 중간 analysis 파일은
의도적으로 Git에서 제외된다. 일반 학습에는 이 절차가 필요 없다.

```bash
python3 post_training/research/data_prep/build_preference_pairs_strict_v2.py
python3 post_training/research/audits/validate_preference_pairs_strict_v2.py
python3 post_training/research/data_prep/build_mpo_token_masks.py \
  --candidates post_training/research/prepared/preference_candidates_strict_v2.jsonl \
  --output post_training/research/prepared/mpo_tokenized_pairs_strict_v2.jsonl \
  --rejections post_training/research/analysis/strict_v2_mpo_token_mask_rejections.jsonl \
  --summary post_training/research/analysis/strict_v2_mpo_token_mask_summary.json \
  --sample-jsonl post_training/research/analysis/strict_v2_mpo_token_mask_sample_10.jsonl \
  --sample-markdown post_training/research/analysis/STRICT_V2_MPO_TOKEN_MASK_SAMPLE_10.md \
  --sample-ids post_training/research/analysis/strict_v2_sample_ids.txt \
  --local-files-only
python3 post_training/research/audits/validate_mpo_token_masks.py \
  --candidates post_training/research/prepared/preference_candidates_strict_v2.jsonl \
  --tokenized post_training/research/prepared/mpo_tokenized_pairs_strict_v2.jsonl \
  --rejections post_training/research/analysis/strict_v2_mpo_token_mask_rejections.jsonl \
  --samples post_training/research/analysis/strict_v2_mpo_token_mask_sample_10.jsonl \
  --summary post_training/research/analysis/strict_v2_mpo_token_mask_summary.json \
  --output post_training/research/analysis/strict_v2_mpo_token_mask_validation_report.json \
  --local-files-only
python3 post_training/research/data_prep/build_mpo_dataset_contract.py
python3 post_training/research/audits/audit_strict_v2_dataset.py
python3 post_training/research/data_prep/finalize_preference_datasets.py
make -C post_training dry-run
```

CPO와 DPO는 각각 별도 run/output/W&B ID를 사용한다.

```bash
make -C post_training dry-run-cpo
make -C post_training smoke-cpo MODEL_DIR=/workspace/models/sft_final
make -C post_training train-cpo MODEL_DIR=/workspace/models/sft_final
```

```bash
make -C post_training dry-run-dpo
make -C post_training smoke-dpo MODEL_DIR=/workspace/models/sft_final
make -C post_training train-dpo MODEL_DIR=/workspace/models/sft_final
```

1 epoch, effective batch 128이면 mPO/CPO/DPO 모두 41 optimizer step이다
(`ceil(5,200 / 128) = 41`). 배포 패키지의 `configs/cpo.yaml`과
`configs/dpo.yaml`의 `freeze_embeddings` 옵션 및
optimizer/scheduler/LR 기본값은 mPO와 동일하다.

## Unsloth hard contract

mPO/CPO 운영 entry point는 다음 값을 강제한다.

```yaml
training:
  backend: unsloth
  freeze_embeddings: false
  logits_projection: selected
  token_logp_backend: unsloth_fused
```

DPO는 여기에 patched trainer와 고정 reference 계약을 추가한다.

```yaml
loss:
  trainer: UnslothDPOTrainer
  reference_free: false
  rpo_alpha: null
  ld_alpha: null
  loss_weights: null
reference:
  mode: initial_sft_policy_precomputed
  precompute_ref_log_probs: true
training:
  use_logits_to_keep: true
  padding_free: false
```

자동 backend 전환, full-sequence logits fallback, PyTorch CE fallback, BF16→FP16 전환은
없다. `FastModel.from_pretrained`에는 `full_finetuning=True`, `return_logits=True`,
`load_in_4bit=False`, `load_in_8bit=False`를 전달한다. 지원하지 않는 인자를 조용히 빼는
호환성 경로도 없다. 또한 import 전에 `UNSLOTH_RETURN_LOGITS=1`과
`UNSLOTH_COMPILE_DISABLE=1`을 설정한다. 후자는 Gemma4 E2B의 non-reentrant activation
checkpoint recomputation이 다른 compiled decoder-layer variant를 선택해 metadata가
달라지는 것을 막는 필수 실행 계약이다. Unsloth loader, Gemma4 patch, fused kernel과
gradient checkpointing은 유지되며 backend fallback은 없다. mPO/CPO의
per-token log-probability는 Unsloth large-vocab fused CE를 사용한다. DPO는 TRL의 표준
completion log-prob 계산을 유지하되 `logits_to_keep`로 prompt 구간을 projection에서
제외한다. 즉 DPO는 completion suffix 전체를 계산하며 term-only mPO projection과는 다르다.

현재 repo pin인 `unsloth==2026.7.2`, `unsloth-zoo==2026.7.2`,
`transformers==5.5.3`, `trl==0.24.0` 소스를 확인한 결과:

- `FastModel.from_pretrained`는 `full_finetuning`과 `return_logits`를 받는다.
- Transformers 5.5의 `Gemma4ForConditionalGeneration.forward`는 tensor `logits_to_keep`를 받고 해당 hidden-state 위치에만 LM head를 적용한다.
- Unsloth compiler는 `return_logits=True`일 때 raw logits 반환을 켠다.
- Unsloth는 TRL의 DPO class와 Gemma4 Processor/collator 경로를 패치하며 text-only
  `images=None`과 `mm_token_type_ids`를 처리한다.

그래도 로컬 macOS 환경에는 Unsloth/CUDA가 없으므로 실제 Gemma4 backward는 외부 GPU image에서 검증해야 한다. 이를 선택 사항으로 두지 않고 one-step receipt를 full run의 hard gate로 만들었다.

GPU 환경은 `requirements-gpu.txt`의 버전을 정확히 사용해야 한다. smoke receipt에는 다음이 모두 들어가며, 하나라도 달라지면 full run이 거부된다.

- post-training 핵심 소스 7개 SHA256 (`mpo_wandb.py` 포함)
- 로컬 `checkpoints/final` 전체 파일의 content SHA256과 `dqs_stage_model.json`
- 데이터 contract와 training-semantic SHA256
- tokenizer vocabulary/backend-core SHA256
- 전체 training config와 loss config
- PyTorch/Unsloth/Transformers 등 runtime version과 CUDA 장치 계약

`model.name_or_path`는 로컬에 내려받은 실제 `checkpoints/final` 디렉터리여야 한다. 존재하지 않는 경로를 Hub ID로 해석하거나 provenance marker가 없는 모델을 쓰는 경로는 허용하지 않는다. marker의 run ID, 최종 subset `22`, global step `184`도 config와 정확히 일치해야 한다.

## 실행 순서

먼저 config의 `model.name_or_path`를 외부 인스턴스에 내려받은 기존 SFT `checkpoints/final` 경로로 바꾼다.

embedding freeze 실험을 할 때만 config를 바꾸거나 모든 실행에 동일한 override를
붙인다. smoke와 full 사이에 이 값이 달라지면 contract mismatch로 거부된다.

```bash
cd post_training
PYTHONPATH=src python3 src/train_mpo.py --config configs/mpo.yaml --dry-run \
  --set model.name_or_path=/workspace/models/sft_final \
  --set training.freeze_embeddings=true
PYTHONPATH=src python3 src/train_mpo.py --config configs/mpo.yaml --smoke-step \
  --set model.name_or_path=/workspace/models/sft_final \
  --set training.freeze_embeddings=true
PYTHONPATH=src python3 src/train_mpo.py --config configs/mpo.yaml \
  --set model.name_or_path=/workspace/models/sft_final \
  --set training.freeze_embeddings=true
```

```bash
# BF16 CUDA GPU와 2.4 <= torch < 2.11인 image에서 실행
cd post_training
pip install -r requirements-gpu.txt
make dry-run MODEL_DIR=/workspace/models/sft_final
make smoke-mpo MODEL_DIR=/workspace/models/sft_final
make train-mpo MODEL_DIR=/workspace/models/sft_final
```

DPO만 실행할 때는 같은 순서에서 entry point만 바꾼다.

```bash
make dry-run-dpo MODEL_DIR=/workspace/models/sft_final
make smoke-dpo MODEL_DIR=/workspace/models/sft_final
make train-dpo MODEL_DIR=/workspace/models/sft_final
```

멀티 GPU이면 smoke와 full을 **같은** world size로 실행한다.

```bash
make smoke-mpo MODEL_DIR=/workspace/models/sft_final \
  LAUNCH='torchrun --nproc_per_node=4'
make train-mpo MODEL_DIR=/workspace/models/sft_final \
  LAUNCH='torchrun --nproc_per_node=4'
```

single-GPU smoke receipt로 multi-GPU full run을 여는 것은 hardware/world-size contract mismatch로 거부된다.

DPO의 reference precompute는 모든 rank에서 실행한다. resume도 매번 SFT 원본을 먼저
로드해 reference를 선계산한 뒤에만 자기 DPO `checkpoint-*`를 restore하므로, 재개된
policy가 reference로 오염되지 않는다.

- `--dry-run`: 모델을 읽지 않고 final source-filtered 5,200행 전체 schema/mask/hash를 검증한다.
- `--smoke-step`: 동일 model/data/loss/code/hardware contract로 실제 Unsloth optimizer step 하나를 실행한다.
- full run: `smoke_step_result.json`의 contract hash와 현재 환경이 정확히 같을 때만 시작한다.

`--smoke-step` 뒤 모델 가중치는 저장하지 않는다. 본 학습은 원래 `checkpoints/final`에서 다시 로드하므로 smoke update가 섞이지 않는다.

## W&B logging 계약

full run은 Transformers의 내장 W&B integration을 쓰지 않는다.
`training.report_to: []`를 유지하고 `mpo_wandb.py`의 strict callback만 사용한다.
내장 callback이 이미 `train/...`인 custom key에 `train/`을 다시 붙이는 문제를 피하고,
W&B import/init/define/log/finish 중 하나라도 실패하면 학습도 실패하게 하기 위해서다.
누락된 패키지나 인증 오류를 경고만 남기고 계속하는 fallback은 없다.

외부 GPU image에는 `requirements-gpu.txt`의 `wandb==0.28.0`이 정확히 설치되어
있어야 하고, full run 전에 `WANDB_API_KEY` 또는 동등한 W&B 인증이 준비되어야 한다.
smoke는 package/version/source contract를 검사하지만 W&B run은 만들지 않는다.

기본 mPO run identity는 다음과 같다. CPO/DPO는 각 config의 독립 ID를 쓴다.

```text
project: dqs
run_id: gemma4_e2b_dqs_mpo_setting5_seed42
group: gemma4_e2b_dqs_mpo_setting5
resume: allow
```

full train, 자기 `checkpoint-*`에서의 resume, 이후 `val`/`final` 평가는 모두 이
run ID를 다시 연다. W&B 로컬 파일은 post-training output의 `wandb/` 아래에만
생긴다. model/checkpoint artifact upload와 parameter watch는 모두 꺼져 있다.

`logging_steps: 1`이므로 single-GPU 1 epoch에서는 마지막 partial accumulation을 포함해
41 optimizer step이 예상된다(`ceil(5,200 / 128) = 41`).
gradient-accumulation micro-step마다 기록하지 않는다. custom loss/margin/token metric은
각 rank에서 `(row-weighted sum, row count)`로 모은 뒤 logging step마다 하나의 DDP
all-reduce를 거쳐 global mean으로 기록한다.

아래 custom loss key는 mPO/CPO에서 기록한다.

- `train/loss/total`, `train/loss/sft_unweighted`, `train/loss/sft_weighted`
- `train/loss/mpo_unweighted`, `train/loss/mpo_weighted`
- `train/margin/mean`, `train/margin/target_gap`, `train/margin/preference_accuracy`
- `train/logp/*`, `train/tokens/*`
- `train/optimizer/learning_rate`, `train/optimizer/grad_norm`
- `train/checkpoint/saved`, `train/status/*`
- `eval/val/*`, `eval/final/*` (final marker의 `train/global_step`에 연결)

DPO는 TRL이 내보내는 `loss`, chosen/rejected reward, log-probability, accuracy 등을
`train/hf/*` 아래에 기록하고, reference 선계산 행 수와 평균/해시는 run manifest에 남긴다.

평가 wrapper는 shared `src/eval.py`의 best-effort W&B 경로를 항상 끄고, 평가가
성공해 `eval_summary.json`을 만든 뒤에만 post-training strict logger로 같은 run에
append한다. 평가에서만 명시적으로 원격 기록을 생략해야 할 때는
`eval_mpo.py --skip-wandb-log`를 사용할 수 있다. 이는 다른 logger로 전환하는
fallback이 아니라 평가 기록 자체의 명시적 opt-out이다.

## checkpoint와 post-training eval 경로

기본값은 final-only가 아니다. 예상 41 optimizer step 동안 `save_steps: 10`,
`save_total_limit: 2`이므로 장애 복구용 `checkpoint-*`는 최근 2개만 남고, 학습이
정상 종료되면 평가 대상으로 쓸 모델을 별도 `final/`에 저장한다.

DPO는
`post_training/outputs/dpo/` 아래에만
`checkpoint-*`, `smoke_step/`, `final/`, `run_manifest.json`을 만들고 최종 모델에는
`dqs_dpo_model.json`을 기록한다. 기존 SFT와 mPO/CPO 출력은 덮어쓰지 않는다.

```text
post_training/outputs/mpo/
├── checkpoint-*       # 최근 2개, 재개용
├── final/             # 유일한 최종 평가 모델
├── eval/
│   ├── val/           # 기존 SFT val과 같은 profile의 비교 평가
│   └── final/         # 필요할 때 최종 test profile 평가
└── smoke_step/
```

평가는 `eval_mpo.py`를 통해 실행한다. 이 entry point는 `final/dqs_mpo_model.json`의
run ID와 setting을 검증하고, 모델 경로를 post-training `final/`로 명시한다. 출력
경로도 동일 post-training run의 `eval/<profile>/`만 허용하므로 기존 SFT의
`artifacts/runs/gemma4_e2b_it_full_iter_lowqe_sf_on_seed42/eval/...`을 읽거나
덮어쓰지 않는다. 모델 자동 탐색 fallback은 사용하지 않는다.

```bash
# 기존 SFT 뒤 실행한 val과 직접 비교
python3 post_training/research/evaluation/eval_mpo.py --profile val

# 최종 test profile
python3 post_training/research/evaluation/eval_mpo.py --profile final
```

중간 `checkpoint-*`까지 없애고 정말 final만 저장하려면 `save_strategy: no`로 바꿀
수 있지만, 실행 중 장애가 나면 재개할 수 없다. 현재 기본값은 경로 충돌과 무관한
복구용 checkpoint 2개를 유지한다.

## Hugging Face dataset 계약

세 objective는 한 공개 dataset repo의 서로 다른 경로에 이미 업로드되어 있다.

```yaml
# release manifest
data_mode: hf
data_access: explicit_download_then_local
hf_dataset:
  repo_id: alwaysgood/dqs-post-training
  revision: 0f7b051f96b3ccdc3837f9537e5aac3a776bf4f1
```

| objective | train file | contract |
|---|---|---|
| mPO | `mpo/train.jsonl` | `mpo/dataset_contract.json` |
| CPO | `cpo/train.jsonl` | `cpo/dataset_contract.json` |
| DPO | `dpo/train.jsonl` | `dpo/dataset_contract.json` |

`make download-data`만 위 remote 정보를 읽는다. 세 기본 config는 각자의
`data/train/*.jsonl`과 bundled contract만 가리킨다. downloader가 raw JSONL SHA256,
row count와 원격 contract hash를 먼저 검사하고, trainer가 schema, tokenizer special
IDs, tokenizer 구현 hash, max length, 실제 학습 tensor의 semantic checksum을 다시
확인한다. 움직일 수 있는 `main`이나 tag는 downloader가 즉시 거부한다.

## 주요 파일

- `configs/mpo_setting5.yaml`: 실행 설정
- `train_mpo.py`: 독립 post-training entry point와 smoke hard gate
- `eval_mpo.py`: final 모델만 허용하는 격리된 post-training eval entry point
- `mpo_objective.py`: setting-5 loss와 mixing coefficient
- `mpo_trainer.py`: chosen/rejected 단일 결합 forward 및 selected logits 계약
- `mpo_masking.py`: padding, causal shift, independent token normalization
- `mpo_data.py`: 명시적으로 내려받은 local JSONL 전용 loader와 전수 검증
- `mpo_model.py`: Unsloth full-tuning loader와 Gemma4 parameter freeze
- `build_preference_pairs_strict_v2.py`: raw golden pairs에서 직접 합성하는 strict builder
- `validate_preference_pairs_strict_v2.py`: raw provenance와 char-span 전수 validator
- `build_mpo_dataset_contract.py`: 최종 token artifact의 immutable contract builder
- `audit_strict_v2_dataset.py`: 알려진 불량/whole-mask/long-span 최종 semantic audit
- `finalize_preference_datasets.py`: 93건 직접 adjudication과 최종 mPO/DPO/CPO 데이터 생성
- `dataset_contract_final_source_filtered.json`: 최종 mPO `train.jsonl` contract
- Hub `dpo/train.jsonl`: DPO용 Teacher-vs-Student pair
- `dataset_contract_full_response_preference.json`: DPO/CPO 공용 pair contract
- `configs/dpo_full_response.yaml`: strict Unsloth full-response DPO 설정
- `train_dpo.py`: patched DPOTrainer, fixed-reference precompute, smoke/final 저장
- `dpo_trainer.py`: processor/tokenization/reference/selected-suffix hard guards
- `configs/cpo_full_response.yaml`: Teacher-vs-Student full-response CPO 설정
- `train_cpo.py`: CPO smoke hard gate와 독립 output/final 저장
- `cpo_objective.py`: sigmoid full-response CPO + Teacher NLL
- `cpo_trainer.py`: Gemma4 completion-only selected-logit CPO trainer
- Hub `cpo/train.jsonl`: runnable CPO train artifact
- `dataset_contract_cpo_full_response.json`: runnable CPO artifact contract
- `requirements-gpu.txt`: source-verified exact Unsloth/Transformers/TRL pins
