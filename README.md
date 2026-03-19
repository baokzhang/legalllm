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
bash scripts/prepare_autodl_env.sh
bash scripts/run_prepare_data.sh
```

If you clone all three repositories as siblings, the scripts also work without
setting these variables.

## AutoDL cache placement

AutoDL system disks are often small. This project defaults model and pip caches
to `/root/autodl-tmp` through [set_cache_env.sh](/Users/zhangbaokun/Documents/DL/LLM/LegalLLM-ClosedLoop/scripts/set_cache_env.sh).

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
