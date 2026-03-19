# LegalLLM Closed Loop

This project turns the existing three repositories in this workspace into a
reproducible legal-domain training pipeline:

- `../HealthAI-2025`: data construction methodology
- `../MedicalGPT`: training framework
- `../lm-evaluation-harness`: evaluation framework

The first milestone focuses on a task that is easy to reproduce with public
Chinese legal data:

- Task 1: criminal charge prediction
- Task 2: case analysis generation with structured JSON output

The default public datasets are:

- `china-ai-law-challenge/cail2018`
- `ShengbinYue/DISC-Law-SFT` (optional SFT augmentation)

## Project layout

```text
LegalLLM-ClosedLoop/
├── configs/
├── data/
├── lm_eval_tasks/
├── requirements-autodl.txt
└── scripts/
```

## What this project actually implements

1. Download public legal datasets to local files.
2. Normalize criminal case records into a structured `feature_content`.
3. Build an aligned training subset using local embedding similarity.
4. Export:
   - ShareGPT-format SFT data for `MedicalGPT`
   - DPO-format preference data for `MedicalGPT`
   - custom `lm-eval` task files for held-out evaluation
5. Provide AutoDL setup and run scripts.

## Quick start

From this directory:

```bash
bash scripts/prepare_autodl_env.sh
bash scripts/run_prepare_data.sh
bash scripts/run_legal_sft.sh
bash scripts/run_merge_adapter.sh outputs/sft-qwen2.5-3b-lora outputs/sft-qwen2.5-3b-merged
bash scripts/run_legal_eval.sh outputs/sft-qwen2.5-3b-merged
```

Then optionally run:

```bash
bash scripts/run_legal_dpo.sh
bash scripts/run_merge_adapter.sh outputs/dpo-qwen2.5-3b-lora outputs/dpo-qwen2.5-3b-merged
bash scripts/run_legal_eval.sh outputs/dpo-qwen2.5-3b-merged
```

## AutoDL split workflow

If you want to boot AutoDL without a GPU first, use this two-stage flow.

No-GPU stage:

```bash
source scripts/set_cache_env.sh
bash scripts/run_bootstrap_no_gpu.sh
```

This stage will:

- create the Python environment
- use a project-local `venv` under `/root/autodl-tmp/envs/legal-llm` by default
- install dependencies
- download public datasets
- build the lightweight case corpus and held-out eval files
- prefetch the base model and embedding model into `/root/autodl-tmp`

After you attach a GPU:

```bash
source scripts/set_cache_env.sh
bash scripts/run_prepare_training_data.sh
bash scripts/run_legal_sft.sh
bash scripts/run_merge_adapter.sh outputs/sft-qwen2.5-3b-lora outputs/sft-qwen2.5-3b-merged
bash scripts/run_legal_eval.sh outputs/sft-qwen2.5-3b-merged
```

For the first SFT run, the default script uses a small eval subset to avoid
spending most of the time on validation. Useful overrides:

```bash
export SFT_DO_EVAL=true
export SFT_MAX_EVAL_SAMPLES=200
export SFT_EVAL_STEPS=500
export SFT_PER_DEVICE_EVAL_BATCH_SIZE=8
```

For `lm-eval`, the default wrapper limits the first run to `1000` examples to
avoid evaluating the full held-out set. Useful overrides:

```bash
export EVAL_LIMIT=1000
export BATCH_SIZE=8
```

If Hugging Face is unstable on AutoDL, you can switch data/model downloads with
environment variables before running the bootstrap:

```bash
export LEGAL_MODEL_SOURCE=modelscope
export LEGAL_CAIL_SOURCE=modelscope
export LEGAL_CAIL_MODELSCOPE_ID=<your_cail_modelscope_dataset_id>
export LEGAL_DISC_SOURCE=modelscope
export LEGAL_DISC_MODELSCOPE_ID=Robin021/DISC-Law-SFT
```

The current default config already uses ModelScope for legal datasets:

- `qazwsxplkj/CAIL2018`
- `Robin021/DISC-Law-SFT`

The current default config also uses ModelScope for model prefetch. The
bootstrap writes local paths to:

- `data/processed/prefetched_model_paths.json`
- `data/processed/prefetched_model_paths.sh`

Later training scripts source that env file automatically, so SFT/DPO/merge use
the local prefetched model path instead of pulling from Hugging Face again.

If you already downloaded CAIL2018 or DISC-Law-SFT somewhere else, the pipeline
also supports local paths:

```bash
export LEGAL_CAIL_LOCAL_DIR=/root/autodl-tmp/manual_datasets/cail2018
export LEGAL_DISC_LOCAL_FILE=/root/autodl-tmp/manual_datasets/DISC-Law-SFT-Pair-QA-released.jsonl
```

## Local workstation -> AutoDL

Recommended approach:

1. Keep the whole `LLM/` directory structure unchanged.
2. Upload or clone this workspace to AutoDL so that these sibling paths still exist:
   - `MedicalGPT`
   - `lm-evaluation-harness`
   - `LegalLLM-ClosedLoop`
3. On AutoDL, `cd LegalLLM-ClosedLoop` and run the commands above.

This project assumes the sibling layout stays the same because the scripts use
relative paths to the training and evaluation repositories.

## Git clone on AutoDL

You can clone this repository alone, then point the scripts to the other two
repositories with environment variables.

Example:

```bash
mkdir -p /root/autodl-tmp/legal-workspace
cd /root/autodl-tmp/legal-workspace

git clone <your-legal-project-repo-url>
git clone https://github.com/shibing624/MedicalGPT.git
git clone https://github.com/EleutherAI/lm-evaluation-harness.git

export MEDICALGPT_ROOT=/root/autodl-tmp/legal-workspace/MedicalGPT
export LMEVAL_ROOT=/root/autodl-tmp/legal-workspace/lm-evaluation-harness

cd /root/autodl-tmp/legal-workspace/LegalLLM-ClosedLoop
source scripts/set_cache_env.sh
bash scripts/run_bootstrap_no_gpu.sh
```

If you clone all three repositories as siblings, the scripts also work without
setting these variables.

## AutoDL cache placement

AutoDL system disks are often small. This project defaults model and pip caches
to `/root/autodl-tmp` through [set_cache_env.sh](/Users/zhangbaokun/Documents/DL/LLM/LegalLLM-ClosedLoop/scripts/set_cache_env.sh).

The environment bootstrap also defaults to `venv` instead of `conda`, which is
more robust on AutoDL images that still carry stale mirror settings in
`/root/.condarc`.

Manual check:

```bash
source scripts/set_cache_env.sh
echo $PIP_CACHE_DIR
echo $HF_HOME
echo $HF_DATASETS_CACHE
```

## Why law is the default choice here

Compared with industrial diagnosis, law is easier to implement end-to-end with
public Chinese data because:

- public datasets are abundant and open
- held-out evaluation is easier to construct
- label spaces such as accusations and law articles are explicit
- preference data can be synthesized with less hidden domain knowledge

Industrial diagnosis can be stronger as a final story, but legal data is much
easier to reproduce on AutoDL without private corpora.
