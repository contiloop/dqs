---
pretty_name: DQS Post-Training Preference Data
language:
  - en
  - ko
task_categories:
  - translation
size_categories:
  - 1K<n<10K
tags:
  - preference-optimization
  - terminology-translation
  - mpo
  - cpo
  - dpo
configs:
  - config_name: mpo
    data_files:
      - split: train
        path: mpo/train.jsonl
  - config_name: cpo
    data_files:
      - split: train
        path: cpo/train.jsonl
  - config_name: dpo
    data_files:
      - split: train
        path: dpo/train.jsonl
---

# DQS Post-Training Preference Data

Strict English-to-Korean preference data for three post-training objectives.
All three configurations contain the same ordered set of 5,200 preference
examples after source-quality review and exclusion of one Teacher/Student pair
with no response-level preference.

## Configurations

| config | chosen / rejected construction | rows | training representation |
|---|---|---:|---|
| `mpo` | Teacher post-edit / the same post-edit with annotated terminology reverted to the Student term | 5,200 | pre-tokenized, independent chosen/rejected term masks |
| `cpo` | complete Teacher response / original complete Student response | 5,200 | pre-tokenized, independent full-completion masks including EOS |
| `dpo` | complete Teacher response / original complete Student response | 5,200 | serialized prompt plus completion strings |

The mPO negative is synthetic only at annotated terminology spans; the rest of
the Teacher post-edit is unchanged. CPO and DPO never use that synthetic
negative: their rejected response is the original full Student output.

## Strict invariants

- No repair or fallback was used during finalization.
- Every retained row has a non-empty preference contrast.
- mPO positive and negative term masks are independently aligned and may have
  different token counts.
- Prompt and padding tokens are excluded from token-level objectives.
- Causal one-token shift is represented in the stored prediction indices.
- CPO and DPO retain the original Teacher/Student responses byte-for-byte.
- Sequence truncation is forbidden by the contracts.
- Completion EOS is `<turn|>` (token id 106), matching the exact final SFT tokenizer; the earlier base-tokenizer EOS id 1 is forbidden.

Each directory contains `train.jsonl` and `dataset_contract.json`. The contract
pins the artifact SHA256, row count, tokenizer revision, tokenizer vocabulary
hash, and objective-specific invariants. `manifest.json` provides a compact
cross-configuration inventory.

## Loading

Always pin an exact 40-character repository commit rather than `main`.

```python
from datasets import load_dataset

dataset = load_dataset(
    "alwaysgood/dqs-post-training",
    "dpo",  # mpo | cpo | dpo
    revision="<40-hex-commit>",
    split="train",
)
```

The mPO and CPO files are already tokenized for the contracted Gemma tokenizer;
do not apply a chat template or retokenize them. The DPO prompt is already a
serialized chat prompt ending at the model prefix; do not apply the chat
template again.

## Artifact hashes

| config | `train.jsonl` SHA256 |
|---|---|
| `mpo` | `a7b7af39b1003619ac6788f18fdfb85e4e0fe76c06ecc8d760f47c8bfe0f339d` |
| `cpo` | `9d9c3e9738059df5f2ceed49b57bc67cc8bc5a23a5e6fa80535447165f2c5f85` |
| `dpo` | `4ff1fe26d35518b4c76ddc50f34ce48def8df73b0f9aec3f61ab97aba00e6187` |

## Provenance

- Source run: `gemma4_e2b_it_full_iter_lowqe_sf_on_seed42`
- Source dataset repository: `alwaysgood/dqs-runs`
- Source dataset commit: `a58b1878988efcecc9a2644f8324bd00131864b5`
- Tokenizer: `google/gemma-4-E2B-it`
- Resolved tokenizer revision: `9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`

- Final SFT tokenizer: `alwaysgood/dqs-runs@a58b1878988efcecc9a2644f8324bd00131864b5`, `gemma4_e2b_it_full_iter_lowqe_sf_on_seed42/checkpoints/final`

No license is asserted by this dataset card. Users are responsible for
complying with the terms applicable to the source content and model/tokenizer.
