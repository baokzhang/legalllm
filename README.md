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

## Project report

### Goal

This project is not just a single fine-tuning run. The goal is to build a
closed loop for a Chinese legal-domain LLM:

- data construction
- supervised fine-tuning
- model merge
- automatic evaluation
- failure analysis
- targeted re-run after fixing implementation issues

The concrete target task is to turn `Qwen2.5-3B-Instruct` into a Chinese
criminal-case assistant that can read case facts and produce strict JSON with:

- `charges`
- `articles`
- `reason`

### Training target

The SFT data exported by this project trains the model to generate structured
legal analysis, not just a single accusation label. A typical target output is:

```json
{
  "charges": "交通肇事",
  "articles": "刑法第133条",
  "reason": "1. ...\n2. ...\n3. ..."
}
```

This design is useful for legal assistant scenarios because downstream users
usually need explicit accusations, law articles, and a readable basis for the
decision instead of a bare class id.

### Evaluation design

To understand what the model actually improves, this project uses two different
evaluation directions.

1. `legal_charge_mc`
   - custom `lm-eval` multiple-choice task
   - given case facts and 6 candidate accusations, choose the most likely one
   - measures accusation classification ability
2. Legal generation evaluation
   - measures whether the model can output usable legal-analysis JSON
   - metrics:
     - `json_valid_rate`
     - `charges_exact_rate`
     - `articles_exact_rate`
     - `articles_contains_gold_rate`
     - `reason_structured_rate`
     - `all_core_fields_correct_rate`

This split turned out to be important because classification and structured
generation do not measure the same capability.

### Key experiments

All numbers below are from a `1000`-example held-out subset used for fast and
reproducible comparison.

#### Classification evaluation: `legal_charge_mc`

| Model | Setting | `acc` | `acc_norm` |
| --- | --- | ---: | ---: |
| Base | `Qwen2.5-3B-Instruct` | `0.521` | `0.016` |
| SFT v1 | first legal SFT run | `0.516` | `0.015` |
| SFT v2 | prompt-fixed run, `checkpoint-1000` | `0.448` | `0.010` |

At first glance, this looks like SFT did not help and even hurt the task.

#### Generation evaluation

| Model | `json_valid` | `charges_exact` | `articles_exact` | `articles_contains_gold` | `reason_structured` | `all_core_fields_correct` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | `0.540` | `0.001` | `0.000` | `0.000` | `0.000` | `0.000` |
| SFT v2 | `0.992` | `0.961` | `0.945` | `0.957` | `0.973` | `0.938` |

This shows that the SFT model learned the structured legal-analysis generation
task very well, even though it did not improve the multiple-choice accusation
classification metric.

### Important debugging result

During the first iteration, the training result on `legal_charge_mc` was almost
flat. The root cause was not immediately visible from the metric itself.

After checking the data export path, training script, and prompt template, this
project found a real implementation mismatch:

- the legal SFT dataset wrote the field `system`
- the training pipeline expected `system_prompt`
- the `qwen` template fell back to the default prompt
- the actual training prompt became `You are a helpful assistant.`

So the first SFT run did not fully inject the intended Chinese legal system
instruction.

This repository then fixed the full prompt path:

- `MedicalGPT/template.py`
- `MedicalGPT/supervised_finetuning.py`
- `scripts/build_legal_sft_dataset.py`
- `scripts/patch_medicalgpt_system_prompt.py`

After the fix, training logs confirmed that the model input started with the
intended legal system block in ChatML format.

### Main conclusion

The most important result of this project is not "SFT failed". The actual
finding is more precise:

- legal SFT did **not** improve the `legal_charge_mc` multiple-choice
  classification task
- legal SFT did **significantly** improve structured legal-analysis generation

In other words, the current training target and the classification metric are
misaligned.

The SFT data optimizes the model toward:

- valid JSON output
- accusation extraction
- law-article alignment
- structured legal reasoning

But `legal_charge_mc` evaluates a narrower capability:

- candidate accusation discrimination

This is why the project needed both evaluation tracks. Without the generation
evaluation, the wrong conclusion would have been that the SFT run was useless.

### What this project demonstrates

This repository now demonstrates a full experimental loop:

1. Build legal-domain data from public Chinese case datasets.
2. Fine-tune a base model with LoRA SFT.
3. Evaluate on a held-out legal classification task.
4. Investigate the failure instead of stopping at the top-line metric.
5. Locate and fix a system-prompt injection bug.
6. Re-run the model and add a second evaluation track aligned with the training
   target.
7. Separate "classification ability" from "structured legal generation ability".

That makes the project useful as both:

- an engineering pipeline for reproducible legal-domain experiments
- a case study in why training objectives and evaluation objectives must be
  aligned

### Current recommendation

If the next goal is better accusation classification, the next iteration should
not keep tuning the current generation-style SFT objective blindly. A better
next step is one of:

- build accusation-classification-oriented SFT data
- train a multi-task setup that includes both accusation classification and
  structured legal generation

If the goal is a usable legal assistant, the current SFT direction is already
validated by the generation metrics above.

## Quick start

From this directory:

```bash
bash scripts/prepare_autodl_env.sh
bash scripts/run_prepare_data.sh
bash scripts/run_legal_sft.sh
bash scripts/run_merge_adapter.sh outputs/sft-qwen2.5-3b-lora outputs/sft-qwen2.5-3b-merged
bash scripts/run_legal_eval.sh outputs/sft-qwen2.5-3b-merged
```

To run the generation-style evaluation aligned with the SFT objective:

```bash
bash scripts/run_legal_generation_eval.sh outputs/sft-qwen2.5-3b-merged
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
