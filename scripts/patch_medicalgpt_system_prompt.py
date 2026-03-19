from __future__ import annotations

import argparse
import os
from pathlib import Path

def resolve_medicalgpt_root(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env_value = os.environ.get("MEDICALGPT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "MedicalGPT"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Unable to locate expected snippet for {label}.")
    return text.replace(old, new, 1)


def patch_template(template_path: Path) -> bool:
    original = template_path.read_text(encoding="utf-8")
    patched = replace_once(
        original,
        (
            "        system_prompt = system_prompt or self.system_prompt\n"
            '        system_prompt = system_prompt + self.sep if system_prompt else ""  # add separator for non-empty system prompt\n'
        ),
        (
            "        system_prompt = system_prompt or self.system_prompt\n"
            "        if system_prompt:\n"
            '            if self.name == "qwen":\n'
            "                stripped_prompt = system_prompt.strip()\n"
            '                if not stripped_prompt.startswith("<|im_start|>system"):\n'
            '                    stripped_prompt = f"<|im_start|>system\\n{stripped_prompt}\\n<|im_end|>"\n'
            "                system_prompt = stripped_prompt + self.sep\n"
            "            else:\n"
            '                system_prompt = system_prompt + self.sep  # add separator for non-empty system prompt\n'
        ),
        "template.py",
    )
    if patched == original:
        return False
    template_path.write_text(patched, encoding="utf-8")
    return True


def patch_sft(sft_path: Path) -> bool:
    original = sft_path.read_text(encoding="utf-8")
    patched = replace_once(
        original,
        (
            "        def get_dialog(examples):\n"
            '            system_prompts = examples.get("system_prompt", "")\n'
        ),
        (
            "        def get_dialog(examples):\n"
            '            system_prompts = examples.get("system_prompt", "")\n'
            '            legacy_system_prompts = examples.get("system", "")\n'
        ),
        "supervised_finetuning.py header",
    )
    patched = replace_once(
        patched,
        (
            "                if not system_prompt:\n"
            '                    system_prompt = system_prompts[i] if system_prompts else ""\n'
            "                yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)\n"
        ),
        (
            "                if not system_prompt:\n"
            "                    if system_prompts:\n"
            "                        system_prompt = system_prompts[i]\n"
            "                    elif legacy_system_prompts:\n"
            "                        # Backward compatibility for ShareGPT files that stored the prompt under `system`.\n"
            "                        system_prompt = legacy_system_prompts[i]\n"
            "                yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)\n"
        ),
        "supervised_finetuning.py fallback",
    )
    if patched == original:
        return False
    sft_path.write_text(patched, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--medicalgpt_root",
        default=None,
        help="Path to the MedicalGPT repository. Defaults to MEDICALGPT_ROOT or ../MedicalGPT.",
    )
    args = parser.parse_args()

    medicalgpt_root = resolve_medicalgpt_root(args.medicalgpt_root)
    template_path = medicalgpt_root / "template.py"
    sft_path = medicalgpt_root / "supervised_finetuning.py"

    if not template_path.exists():
        raise FileNotFoundError(f"template.py not found: {template_path}")
    if not sft_path.exists():
        raise FileNotFoundError(f"supervised_finetuning.py not found: {sft_path}")

    changed_template = patch_template(template_path)
    changed_sft = patch_sft(sft_path)

    print(f"MedicalGPT root: {medicalgpt_root}")
    print(f"Patched template.py: {changed_template}")
    print(f"Patched supervised_finetuning.py: {changed_sft}")


if __name__ == "__main__":
    main()
