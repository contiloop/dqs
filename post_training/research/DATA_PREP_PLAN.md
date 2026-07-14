# strict-v2 + source review 최종 preference 데이터 계약

## 목적

논문에서는 mPO의 term-token preference loss를 가져오고, pair 합성은 DQS
`golden_pairs` 구조에 맞게 별도로 수행한다.

```text
y+ = teacher의 최종 post-edit(target)
y- = y+의 terminology correction span을 student term으로 되돌린 결과
```

- 한 행의 terminology annotation이 여러 개면 전부 한 pair에 공동 역치환한다.
- 한 annotation이라도 안전하지 않으면 일부만 쓰지 않고 행 전체를 제외한다.
- 선택된 span 밖의 문자열은 `y+`와 `y-`에서 byte-for-byte 동일해야 한다.
- `M+`와 `M-`는 각 completion에서 독립적으로 만든다.
- 합성, token 정렬, 학습 어느 단계에도 repair/fuzzy/fallback은 없다.

## raw부터 다시 만든 canonical 결과

strict-v2 builder는 기존 7,835/6,714/5,888행 artifact를 입력으로 읽지 않는다.
`post_training/research/raw/golden_pairs/subset_000.jsonl`부터 `subset_022.jsonl`까지를
source of truth로 다시 읽는다.

| 단계 | rows |
|---|---:|
| raw golden rows | 23,000 |
| terminology 없는 rows | 13,948 |
| terminology rows | 9,052 |
| base exact-mapping reject | 1,217 |
| strict synthesis reject | 1,624 |
| strict char-span candidates | **6,211** |
| tokenizer boundary reject | 706 |
| strict-v2 token rows / source review 입력 | **5,505** |

source review 전 5,505행에는 7,289개 replacement span이 있다. Teacher label 분포는
`minor 3,376`, `major 1,649`, `critical 480`이다. annotation이 2개 이상인 행은
1,254개, repeated occurrence를 공동 역치환한 행은 204개, `y-`가 원래 Student
출력과 같은 roundtrip 행은 1,372개다.

### base exact mapping

먼저 모든 terminology annotation에 대해 다음을 증명한다.

1. `error_span_target`과 `correction`이 비어 있지 않다.
2. NFC 기준 student/teacher term이 실제로 다르다.
3. student term은 원래 student 출력에, teacher term은 teacher target에 exact match다.
4. teacher occurrence가 하나면 그 span을 사용한다.
5. teacher occurrence가 여러 개면 student/teacher occurrence 수가 같은 경우에만 전부
   되돌린다.
6. 여러 teacher span은 겹치지 않는다.
7. 기록된 rejected span을 teacher term으로 복구하면 `y+`가 정확히 재현된다.

base reject 1,217행의 primary 사유는 teacher term 없음 755, 반복 occurrence 모호
344, student term 없음 54, 필수 필드 누락 24, teacher span 겹침 20, NFC 동일 20이다.

### strict-by-construction 검사

base mapping 직후, artifact에 쓰기 전에 같은 행에서 다음을 검사한다.

- 기존 `quality_flags`는 종류와 관계없이 모두 hard reject한다. 따라서 source exact
  mismatch, 조사/구두점/parenthetical 경고뿐 아니라 64자 또는 whitespace token
  8개를 넘는 source/student/teacher span도 남지 않는다.
- Teacher 문맥과 student term 접합으로 새로 생긴 문자열 overlap 및 lexical token
  중복을 reject한다.
- 서로 다른 두 annotated term이 인접해서 생긴 반복과 Student 원출력에 이미 존재한
  반복은 splice defect가 아니므로 허용한다.
- `UTB)의의`, `건너뛰는을`, `기준으로에` 같은 조사·어미 충돌을 reject한다.
  `에서는`, `에서도`처럼 가능한 조사 결합은 허용한다.
- source가 balanced인데 chosen이 unbalanced이거나, 치환으로 괄호·인용부호 balance가
  달라지면 reject한다.
- source/chosen/rejected의 75% 이상을 덮는 유한문장형 annotation을 reject한다.
- chosen/rejected completion 전체를 term span이 덮는 nominal title도 reject한다.

이 단계는 값을 고쳐 살리는 필터가 아니다. raw terminology row 하나가 완전한 pair가
되거나 rejection ledger 한 행이 된다.

## Gemma token mask

char-span candidates는 `google/gemma-4-E2B-it` tokenizer revision
`9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`의 vocabulary/backend로 변환한다.
최종 release에서는 같은 vocabulary/backend를 가진 실제 SFT final tokenizer의 special-token
profile을 적용해 completion EOS를 `<turn|>`(id 106)으로 고정한다. 최초 base-tokenizer
산출물의 `<eos>`(id 1)는 `retarget_preference_eos.py`가 전 행의 마지막 appended EOS에서만
106으로 바꾸며, 본문 token·길이·mask는 바꾸지 않는다.

- prompt와 completion을 각각 `add_special_tokens=False`로 tokenize한다.
- completion 끝에는 EOS를 붙인다.
- prompt와 padding은 모든 loss mask에서 0이다.
- EOS는 SFT completion mask에는 포함하고 term mask에서는 제외한다.
- (M^+), (M^-)는 각자의 char offset으로 독립 정렬한다.
- term token이 비용어 문자를 함께 포함하면 mask를 넓히지 않고 행을 reject한다.
- truncation하지 않으며 `max_seq_length=2,908`을 넘으면 reject한다.
- causal loss는 `logits[:, :-1]`와 `input_ids[:, 1:]`, 즉 `mask[:, 1:]`로 정렬한다.
- completion, chosen term, rejected term loss는 각 행에서 각자의 token 수로 정규화한다.

| token 결과 | 값 |
|---|---:|
| accepted rows | 5,505 |
| chosen term tokens | 43,690 |
| rejected term tokens | 41,651 |
| (M^+), (M^-) token 수가 다른 rows | 4,327 |
| chosen boundary reject | 560 |
| rejected boundary reject | 146 |
| 빈 mask | 0 |
| completion 전체 term mask | 0 |

## source 원문 직접 review와 최종 분할

strict-v2 token 5,505행의 source를 전부 판정했다. 저장된 두 judgment journal의 초기
분포는 `KEEP 5,196 / REJECT 216 / REVIEW 93`이다. REVIEW 93건은 원문 전체를 직접
읽고, 제목-본문 또는 완결된 문서 블록 경계만 명백히 복원 가능한 5건을 KEEP으로
승격했다. 나머지 88건은 절단, 표 구조 손실, 문맥 혼합 또는 불명확한 field association
때문에 제외했다.

| 최종 source 결정 | rows |
|---|---:|
| KEEP | **5,201** |
| REJECT | 304 |
| REVIEW | 0 |

source-quality KEEP 5,201건 중 raw Teacher/Student 응답이 NFC 기준 동일하면서
terminology annotation만 남은 불일치 1건은 faithful preference가 성립하지 않으므로
mPO에서도 fail-closed한다. 최종 mPO는 **5,200건**이다. 별도 DPO/CPO 공용 데이터도
동일한 5,200건에서 `chosen=raw target`, `rejected=raw student_translation`으로 만든다.
이 full-response 데이터에는 mPO의 합성 minimal negative가 들어가지 않는다.

같은 5,200건을 strict CPO 실행용으로 다시 tokenize한다. prompt와 각 completion의
tokenizer boundary가 additive인지 전수 확인하고, Teacher/Student completion 전체와
EOS를 독립 mask로 만든다. 최대 Teacher sequence는 2,794, 최대 Student sequence는
2,825 tokens이며 truncation은 0건이다. CPO artifact의 `term_mask` 필드명은 shared
loader 재사용을 위한 alias일 뿐이고 의미는 completion 전체 mask다.

DPO는 공용 raw `prompt/chosen/rejected`를 `trl==0.24.0`의 Unsloth-patched
`DPOTrainer`에 전달한다. Trainer가 준비한 prompt/Teacher/Student token IDs를 raw 문자열의
기대 ID와 전 행 대조하고, Gemma4 Processor의 text-only all-zero
`mm_token_type_ids`도 검사한다. reference는 별도 full-model clone이 아니라 최초 SFT
policy의 completion log-prob을 DPO update 및 resume restore 전에 선계산한 고정값이다.
RPO/SFT mixing, reference-free, truncation, bare-tokenizer 및 pristine TRL fallback은 없다.

## 최종 감사

- raw terminology 9,052행은 accepted 6,211행과 synthesis rejection 2,841행으로
  정확히 분할된다.
- accepted row의 source, original student, teacher target은 raw와 byte-for-byte 같다.
- 모든 term annotation이 포함되고 비용어 구간은 chosen/rejected에서 동일하다.
- 이전 감사에서 확인한 경계 중복, 조사 충돌, delimiter 오류, 잘못된 teacher 닫는 괄호
  34행은 모두 최종 데이터에서 빠졌다.
- 기존 5,888행에서 남았던 long-span warning 326행과 completion 전체 mask 27행도
  최종 데이터에서 빠졌다.
- 정규화하면 띄어쓰기·하이픈·구분기호만 다른 13행은 Teacher가 실제 terminology
  correction으로 표시했고 양쪽 masked token IDs가 다르므로 유지한다.
- 전체 post-training test suite와 raw/char/token/hash validator, final semantic audit,
  mPO/CPO/DPO trainer dry-run을 실행한다.

고정 checksum은 다음과 같다.

```text
raw manifest SHA256: 43fdccfcd9d2d2858cd4894084c0b7abf0ac4afc3e2ac9fd6772b6185a847eaf
candidate SHA256:    fc79519ab66801247c2c3336cf818117ff02363baafce15a127a763321c98cd8
train artifact SHA:  e5512852a8db36597ef9cf5e080d745b57c25650a65dcb3a225d0a6ecfa1f463
semantic SHA256:     78ee0596b0f312e8369944f62d75ba16224b81b74087b8989f062e613e8132ef

final mPO artifact:  a7b7af39b1003619ac6788f18fdfb85e4e0fe76c06ecc8d760f47c8bfe0f339d
final mPO semantic:  7b7d18476c75ed579067280b89406151aee78a68a2ff0033ea076461c90b5cae
full raw pair:       4ff1fe26d35518b4c76ddc50f34ce48def8df73b0f9aec3f61ab97aba00e6187
CPO train artifact:  9d9c3e9738059df5f2ceed49b57bc67cc8bc5a23a5e6fa80535447165f2c5f85
CPO semantic:        19fc8ba480a2321db1ba2542a0525be1dfca62a7ddf0832e432d69794e6463fa
```

## canonical 산출물과 로컬 보존 경계

최종 train JSONL은 `alwaysgood/dqs-post-training`의 exact commit
`0f7b051f96b3ccdc3837f9537e5aac3a776bf4f1`에 보존한다.

| objective | Hub train file | local contract |
|---|---|---|
| mPO | `mpo/train.jsonl` | `post_training/research/contracts/dataset_contract_final_source_filtered.json` |
| CPO | `cpo/train.jsonl` | `post_training/research/contracts/dataset_contract_cpo_full_response.json` |
| DPO | `dpo/train.jsonl` | `post_training/research/contracts/dataset_contract_full_response_preference.json` |

아래 `prepared/`, `raw/`, `source_quality/` 경로는 재생성 시에만 생기는 로컬 산출물이며
현재 repo에는 보존하지 않는다. 최종 contract, 검증 보고서, human adjudication ledger,
샘플 10건만 Git 대상이다.

- builder: `post_training/research/data_prep/build_preference_pairs_strict_v2.py`
- builder tests: `post_training/research/tests/test_build_preference_pairs_strict_v2.py`
- char candidates: `post_training/research/prepared/preference_candidates_strict_v2.jsonl`
- roundtrip subset: `post_training/research/prepared/roundtrip_strict_v2.jsonl`
- synthesis ledger: `post_training/research/analysis/strict_v2_synthesis_rejections.jsonl`
- synthesis summary: `post_training/research/analysis/strict_v2_synthesis_summary.json`
- raw/char validator: `post_training/research/audits/validate_preference_pairs_strict_v2.py`
- char validation report: `post_training/research/analysis/strict_v2_synthesis_validation_report.json`
- token builder: `post_training/research/data_prep/build_mpo_token_masks.py`
- strict-v2 source-review input: `post_training/research/prepared/mpo_tokenized_pairs_strict_v2.jsonl`
- token rejection ledger: `post_training/research/analysis/strict_v2_mpo_token_mask_rejections.jsonl`
- token summary: `post_training/research/analysis/strict_v2_mpo_token_mask_summary.json`
- token validator report: `post_training/research/analysis/strict_v2_mpo_token_mask_validation_report.json`
- final audit: `post_training/research/audits/audit_strict_v2_dataset.py`,
  `post_training/research/analysis/strict_v2_final_audit_report.json`
- contract builder: `post_training/research/data_prep/build_mpo_dataset_contract.py`
- source review finalizer: `post_training/research/data_prep/finalize_preference_datasets.py`
- final decisions: `post_training/research/source_quality/gpt54mini_source_integrity_v1/final_decisions.jsonl`
- human REVIEW ledger: `post_training/research/analysis/source_quality_human_adjudications.jsonl`
- final mPO candidates: `post_training/research/prepared/preference_candidates_final_source_filtered.jsonl`
- final mPO train file: `post_training/research/prepared/mpo_tokenized_pairs_final_source_filtered.jsonl`
- final mPO contract: `post_training/research/contracts/dataset_contract_final_source_filtered.json`
- full-response DPO/CPO pairs: `post_training/research/prepared/full_response_preference_pairs_final.jsonl`
- full-response DPO/CPO contract: `post_training/research/contracts/dataset_contract_full_response_preference.json`
- DPO config source: `post_training/research/configs/dpo_full_response.yaml`
- DPO entry point: `post_training/dqs_preference_training_hf/src/train_dpo.py`
- DPO runtime guards: `post_training/dqs_preference_training_hf/src/dpo_trainer.py`
- CPO tokenized train file: `post_training/research/prepared/cpo_tokenized_full_response_pairs_final.jsonl`
- CPO tokenized contract: `post_training/research/contracts/dataset_contract_cpo_full_response.json`
- review samples: `post_training/research/analysis/STRICT_V2_SAMPLE_10.md`,
  `post_training/research/analysis/STRICT_V2_MPO_TOKEN_MASK_SAMPLE_10.md`

기존 v1 데이터 artifact는 삭제했다. 비교용 builder 코드는 남아 있지만 현재 training
config는 읽지 않는다.

## 재생성 순서

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
make -C post_training/dqs_preference_training_hf dry-run
make -C post_training test
```

Hub 업로드는 완료되었고 배포 manifest가 위 exact 40-hex commit을 고정한다.
`make download-data`가 세 artifact를 `data/train/`에 먼저 설치한 뒤 config는 local
JSONL만 읽는다. mPO, CPO, DPO artifact는 목적과 negative 정의가 다르므로 각자의
train/contract 경로를 사용하며 한 split으로 합치지 않는다.
