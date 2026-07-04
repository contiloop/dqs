from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class PromptError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    text: str
    template_id: str
    template_group: str
    template_hash: str
    chat_template_applied: bool


def stable_template_index(sample_key: str, template_count: int, seed: int) -> int:
    if template_count <= 0:
        raise PromptError("template_count must be positive")
    digest = hashlib.blake2b(f"{seed}|{sample_key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % template_count


def load_student_templates(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise PromptError(f"student template file must be a mapping: {path}")
    return payload


def _weighted_variant(values: list[Mapping[str, Any]], *, key: str, seed: int) -> str:
    if not values:
        return ""
    rng = random.Random(f"{seed}|{key}")
    total = sum(float(item.get("weight", 0.0) or 0.0) for item in values)
    if total <= 0:
        value = values[0].get("value", "")
        return str(value)
    pick = rng.random() * total
    cursor = 0.0
    for item in values:
        cursor += float(item.get("weight", 0.0) or 0.0)
        if pick <= cursor:
            return str(item.get("value", ""))
    return str(values[-1].get("value", ""))


def _template_values(template_cfg: Mapping[str, Any], *, source: str, row_id: str, seed: int) -> dict[str, Any]:
    language = template_cfg.get("language", {})
    domain = template_cfg.get("domain", {})
    if not isinstance(language, Mapping) or not isinstance(domain, Mapping):
        raise PromptError("student templates require language and domain mappings")
    src = language.get("src", {})
    tgt = language.get("tgt", {})
    if not isinstance(src, Mapping) or not isinstance(tgt, Mapping):
        raise PromptError("student templates require language.src and language.tgt mappings")
    ko_variants = tgt.get("ko_name_variants", [])
    if not isinstance(ko_variants, list):
        ko_variants = []
    tgt_lang_ko = _weighted_variant(ko_variants, key=row_id, seed=seed) or str(tgt.get("name_ko", "한국어"))
    return {
        "src": source,
        "source": source,
        "src_label_en": str(src.get("label_en", "English")),
        "src_label_ko": str(src.get("label_ko", "영어")),
        "tgt_label_en": str(tgt.get("label_en", "Korean")),
        "tgt_label_ko": str(tgt.get("label_ko", "한국어")),
        "src_lang_en": str(src.get("name_en", "English")),
        "src_lang_name": str(src.get("name_en", "English")),
        "src_lang_ko": str(src.get("name_ko", "영어")),
        "tgt_lang_en": str(tgt.get("name_en", "Korean")),
        "tgt_lang_name": str(tgt.get("name_en", "Korean")),
        "tgt_lang_ko": tgt_lang_ko,
        "domain_en": str(domain.get("name_en", "economic and financial")),
        "domain_ko": str(domain.get("name_ko", "경제·금융")),
    }


def _template_group_for_model(model_cfg: Mapping[str, Any]) -> str:
    if bool(model_cfg.get("use_hf_chat_template", False)):
        return "instruct_templates"
    return "base_templates"


def _apply_chat_template(
    *,
    user_prompt: str,
    system_prompt: str,
    model_cfg: Mapping[str, Any],
) -> tuple[str, bool]:
    if not bool(model_cfg.get("use_hf_chat_template", False)):
        if system_prompt:
            return f"{system_prompt.strip()}\n\n{user_prompt}", False
        return user_prompt, False
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise PromptError("transformers is required to apply the HF chat template") from exc

    model_name = str(model_cfg.get("name_or_path", "")).strip()
    if not model_name:
        raise PromptError("model.name_or_path is required to apply the HF chat template")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=str(model_cfg.get("tokenizer_revision", "main")),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt})
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    text = str(rendered)
    text = re.sub(r"\n<think>\s*</think>\s*$", "\n", text)
    for suffix in ("\n<think>\n", "\n<think>\n\n"):
        if text.endswith(suffix):
            text = text[: -len(suffix)] + "\n"
            break
    return text, True


def render_student_prompt(
    *,
    template_cfg: Mapping[str, Any],
    prompt_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    source: str,
    row_id: str,
    subset_idx: int,
) -> RenderedPrompt:
    group_name = _template_group_for_model(model_cfg)
    templates = template_cfg.get(group_name, [])
    if not isinstance(templates, list) or not templates:
        raise PromptError(f"student templates missing non-empty {group_name}")

    seed = int(prompt_cfg.get("template_seed", 42))
    selection = prompt_cfg.get("student_selection", {})
    deterministic = bool(
        isinstance(selection, Mapping) and selection.get("deterministic_by_sample_id", True)
    )
    sample_key = row_id if deterministic else f"{row_id}:{subset_idx}"
    template_idx = stable_template_index(sample_key, len(templates), seed)
    template = templates[template_idx]
    if not isinstance(template, Mapping):
        raise PromptError(f"invalid student template at {group_name}[{template_idx}]")

    text = str(template.get("text", ""))
    system_text = str(template.get("system", ""))
    user_text = str(template.get("user", text))
    template_id = str(template.get("id", f"{group_name}_{template_idx:03d}"))
    values = _template_values(template_cfg, source=source, row_id=row_id, seed=seed)
    rendered_system = system_text.format(**values)
    rendered_user = user_text.format(**values)
    final_prompt, chat_applied = _apply_chat_template(
        user_prompt=rendered_user,
        system_prompt=rendered_system,
        model_cfg=model_cfg,
    )
    hash_source = f"system:{system_text}\nuser:{user_text}"
    template_hash = hashlib.sha256(hash_source.encode("utf-8")).hexdigest()
    return RenderedPrompt(
        text=final_prompt,
        template_id=template_id,
        template_group=group_name,
        template_hash=template_hash,
        chat_template_applied=chat_applied,
    )
