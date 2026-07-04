# DQS

## Quick start

### 1. Clone repository

```sh
git clone <REPO_URL>
cd DQS
```

### 2. Install dependencies

```sh
make set
```

Runtime version notes:

- Python: `3.11`
- CUDA wheel index: `https://download.pytorch.org/whl/cu128`
- Torch stack: `torch==2.10.0`, `torchvision==0.25.0`, `torchaudio==2.10.0`
- vLLM: `vllm==0.19.1`
- Unsloth stack: `unsloth==2026.5.2`, `unsloth-zoo==2026.5.1`
- HF training stack: `transformers==5.5.0`, `trl==0.24.0`, `datasets==3.4.1`
- Hugging Face transfer stack: `huggingface_hub>=1.14.0,<2`, `hf-xet>=1.5.0,<2`
- FlashAttention2: `flash-attn==2.8.3`
- NumPy: `numpy==2.2.6`

### 3. Configure access

```sh
python -c "from huggingface_hub import login; login()"
wandb login

export GEMINI_API_KEY="..."
```

Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` only when their provider weights
are enabled in `configs/teacher.yaml`.

### 4. Validate setup

```sh
make validate-setup
```

The default teacher provider should print as:

```text
gemini model=gemini-3.1-flash-lite weight=1.0 api_key_env=GEMINI_API_KEY
```

### 5. Prepare data

Download the prepared Qwen3.5 1280-token corpus:

```sh
make download-prepared-data
```

Optional: rebuild prepared data from the raw corpus with tokenizer-based
splitting.

```sh
make preprocess-raw PREPROCESS_TOKENIZER_MODEL=unsloth/Qwen3.5-4B
```

### 6. Train

Run one subset:

```sh
make train
```

Run every subset in order:

```sh
make train-stage
```

Create deterministic smoke datasets first:

```sh
make smoke-data
```

This writes:

```text
data/smoke/max_context_sft.jsonl
data/smoke/cycle.jsonl
data/smoke/val.jsonl
data/smoke/smoke_stats.json
```

Check the max-context token budget:

```sh
python - <<'PY'
import json
stats = json.load(open("data/smoke/smoke_stats.json"))
print("max_seq_length", stats["max_seq_length"])
for row in stats["max_context_sft"]:
    print(row["id"], row["template_id"], row["total_token_count"], row["remaining_tokens"])
PY
```

Run the max-context SFT smoke to check training VRAM. This uses rows near the
configured `training.max_seq_length` and runs one SFT step by default:

```sh
make smoke-sft-max-context
```

Run the small end-to-end cycle smoke. This uses two 4-row subsets and exercises
student inference, filtering, teacher generation, SFT, and smoke eval:

```sh
make smoke-cycle
```

For wiring-only cycle validation without model/API calls:

```sh
make smoke-cycle SMOKE_CYCLE_DRY_RUN=1 SMOKE_CYCLE_EVAL_DRY_RUN=1
```

To test multiple training GPUs in the max-context SFT smoke:

```sh
make smoke-sft-max-context SFT_NPROC_PER_NODE=4
```

By default, `make train-stage` runs validation eval after every subset. To use
a wider cadence:

```sh
make train-stage EVAL_EVERY_N_SUBSETS=5
```

Run SFT from an existing subset artifact:

```sh
make sft SFT_SUBSET_IDX=0
```

Run multi-GPU SFT:

```sh
make sft SFT_SUBSET_IDX=0 SFT_NPROC_PER_NODE=4
```

Resume from a specific phase:

```sh
make train TRAIN_START_FROM=teacher
make train TRAIN_START_FROM=sft
```

By default, `make train` uses `TRAIN_RESUME=auto`. If a previous run stopped
mid-subset, it reads `phase_state.json`, selects the latest incomplete subset,
and resumes from the recorded phase.

`make train-stage` uses the same resume state for each subset. Completed
subsets are skipped, and the next incomplete subset resumes from its recorded
phase. `TRAIN_STAGE_END_SUBSET` is exclusive.

During `make train-stage`, SFT uses one LR schedule across the full subset
cycle. Optimizer, scheduler, and global step state are saved at the end of each
subset.

To ignore automatic resume selection, set `TRAIN_RESUME=none`.

Valid resume phases:

```text
input
student-infer
student-filter
qe-select
teacher
sft-dataset
sft
```

Skipped phases must already have their required artifacts under the run
directory, otherwise the command stops with `cannot resume`.

Run a student-filter ablation:

```sh
make train TRAIN_OVERRIDES='data.degeneration_filter.student_enabled=false'
```

The run id includes `sf_on` or `sf_off`, so filter ablation artifacts are
written to separate run directories automatically.

Useful training overrides:

```sh
make train TRAIN_OVERRIDES='model=qwen35_4b_base'
make train TRAIN_OVERRIDES='training=full'
make train TRAIN_OVERRIDES='run.seed=7'
```

Subset artifacts are written under:

```text
artifacts/runs/${run.id}/subsets/subset_000/
```

Key subset files:

```text
front_stage_summary.json
student_filter_summary.json
teacher_summary.json
selected_for_teacher.jsonl
golden_pairs.jsonl
sft_train.jsonl
```

### 7. Evaluate

Run the validation evaluation profile on `data/val.jsonl`:

```sh
make eval EVAL_PROFILE=val
```

Run the heavyweight final evaluation profile on `data/test.jsonl`:

```sh
make eval EVAL_PROFILE=final
```

Run final evaluation for every saved checkpoint:

```sh
make eval-checkpoints
```

To evaluate a downloaded local model artifact:

```sh
make eval EVAL_PROFILE=final EVAL_MODEL_PATH=/path/to/model
```

`Unbabel/XCOMET-XXL` requires accepted Hugging Face access. For MetricX, use a
separate environment and point `METRICX_REPO_DIR` at the cloned repo:

```sh
[ -d /workspace/metricx/.git ] || git clone https://github.com/google-research/metricx /workspace/metricx
python -m venv /root/.venvs/metricx
/root/.venvs/metricx/bin/python -m pip install -U pip
/root/.venvs/metricx/bin/python -m pip install -r /workspace/metricx/requirements.txt

METRICX_PYTHON=/root/.venvs/metricx/bin/python \
METRICX_REPO_DIR=/workspace/metricx \
make eval EVAL_PROFILE=final
```

Eval artifacts are written under:

```text
artifacts/runs/${run.id}/eval/${eval.profile}/
```

Checkpoint eval artifacts are written under:

```text
artifacts/runs/${run.id}/eval/final_by_checkpoint/checkpoint-000008/
artifacts/runs/${run.id}/eval/final_by_checkpoint/summary.jsonl
```

Key eval files:

```text
eval_summary.json
eval_scores.jsonl
eval_filtered.jsonl
eval_translations.jsonl
```

W&B logs compact curves only: SFT loss/LR from Trainer plus subset summary
counts and eval metric means. Row-level requests, outputs, translations, and
scores stay in local JSONL artifacts.

### 8. Upload run artifacts

Upload a completed run folder to a Hugging Face dataset repo:

```sh
make upload-run HF_DATASET_REPO=<HF_ID>/dqs-runs UPLOAD_RUN_ID=<RUN_ID>
```

This uploads the full folder below:

```text
artifacts/runs/<RUN_ID>/
```

By default it is stored under `<RUN_ID>/` inside the dataset repo. To check the
file list first:

```sh
make upload-run HF_DATASET_REPO=<HF_ID>/dqs-runs UPLOAD_RUN_ID=<RUN_ID> UPLOAD_DRY_RUN=1
```

If the dataset repo does not exist, it is created as private by default. Use
`UPLOAD_PRIVATE=0` only for a public repo.

For large uploads:

```sh
HF_XET_HIGH_PERFORMANCE=1 make upload-run HF_DATASET_REPO=<HF_ID>/dqs-runs UPLOAD_RUN_ID=<RUN_ID>
```
