from __future__ import annotations

import re
from typing import Any, Mapping


_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_LATIN_EXT_RE = re.compile(r"[\u00C0-\u024F]")
_CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]")
_KANA_RE = re.compile(r"[\u3040-\u30FF]")
_OTHER_SCRIPT_RE = re.compile(
    r"[\u0400-\u04FF\u0600-\u06FF\u0E00-\u0E7F\u0900-\u097F\u0370-\u03FF]"
)
_WORD_RE = re.compile(r"[A-Za-z0-9가-힣\u00C0-\u024F]+")
_CLAUSE_SPLIT_RE = re.compile(r"[.!?。！？\n;；,，]+")
_PUNCT_SPACE_RE = re.compile(r"[^0-9A-Za-z가-힣\u00C0-\u024F]+")
_SENTENCE_FINAL_RE = re.compile(r"[.!?。！？…\"')\]”’]$")
_THINKING_TRACE_RE = re.compile(
    r"<\s*/?\s*think\s*>|\bwe need to translate\b|\blet'?s understand\b|\bpossible translation\b",
    re.IGNORECASE,
)
_CHAR_REPEAT_PATTERNS: dict[int, re.Pattern[str]] = {}
_ENGLISH_SENTENCE_MARKERS = {
    "am",
    "are",
    "be",
    "been",
    "being",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "was",
    "were",
    "will",
    "would",
}
_KO_FUNCTION_TOKENS = {"의", "및", "등", "또는", "그리고", "거나", "이나", "하며"}
_IGNORED_SHORT_NGRAMS = {
    ("할", "수"),
    ("수", "있습니다"),
    ("수", "있다"),
    ("수", "있는"),
    ("될", "수"),
    ("있을", "수"),
}


def _ratio(count: int, total: int) -> float:
    return float(count) / max(total, 1)


def _script_counts(text: str) -> dict[str, int]:
    return {
        "hangul": len(_HANGUL_RE.findall(text)),
        "latin": len(_LATIN_RE.findall(text)),
        "latin_ext": len(_LATIN_EXT_RE.findall(text)),
        "cjk": len(_CJK_RE.findall(text)),
        "kana": len(_KANA_RE.findall(text)),
        "other": len(_OTHER_SCRIPT_RE.findall(text)),
    }


def _cfg_bool(config: Mapping[str, Any], key: str, default: bool = True) -> bool:
    return bool(config.get(key, default))


def _nested_config(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _cfg_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    return int(value) if isinstance(value, int | float | str) and str(value).strip() else default


def _cfg_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    return float(value) if isinstance(value, int | float | str) and str(value).strip() else default


def _bounded_tokens(text: str, max_tokens: int) -> list[str]:
    tokens = [token.lower() for token in _WORD_RE.findall(text)]
    if len(tokens) <= max_tokens:
        return tokens
    half = max_tokens // 2
    return tokens[:half] + tokens[-half:]


def _token_overlap(source: str, mt: str) -> float:
    src_tokens = {token.lower() for token in _WORD_RE.findall(source) if len(token) >= 3}
    mt_tokens = {token.lower() for token in _WORD_RE.findall(mt) if len(token) >= 3}
    if not src_tokens:
        return 0.0
    return len(src_tokens & mt_tokens) / len(src_tokens)


def _has_english_sentence_marker(text: str) -> bool:
    tokens = {token.lower() for token in _WORD_RE.findall(text)}
    return bool(tokens & _ENGLISH_SENTENCE_MARKERS)


def _max_char_repeat(text: str, max_unit: int = 8, max_chars: int = 12000) -> int:
    if len(text) > max_chars:
        half = max_chars // 2
        text = text[:half] + text[-half:]
    best = 1
    for unit_len in range(1, max_unit + 1):
        pattern = _CHAR_REPEAT_PATTERNS.get(unit_len)
        if pattern is None:
            pattern = re.compile(r"(.{%d})(?:\1)+" % unit_len, re.S)
            _CHAR_REPEAT_PATTERNS[unit_len] = pattern
        for match in pattern.finditer(text):
            unit = match.group(1)
            if not unit.strip():
                continue
            run = len(match.group(0)) // unit_len
            best = max(best, run)
    return best


def _max_word_repeat_tokens(tokens: list[str]) -> int:
    best = 1
    last = None
    run = 0
    for token in tokens:
        if token == last:
            run += 1
        else:
            last = token
            run = 1
        best = max(best, run)
    return best


def _max_token_frequency(tokens: list[str]) -> int:
    counts: dict[str, int] = {}
    best = 0
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
        best = max(best, counts[token])
    return best


def _max_korean_function_token_frequency(tokens: list[str]) -> int:
    counts: dict[str, int] = {}
    best = 0
    for token in tokens:
        if token not in _KO_FUNCTION_TOKENS:
            continue
        counts[token] = counts.get(token, 0) + 1
        best = max(best, counts[token])
    return best


def _max_word_repeat(text: str) -> int:
    return _max_word_repeat_tokens([token.lower() for token in _WORD_RE.findall(text)])


def _immediate_span_repeat(tokens: list[str], max_span: int) -> tuple[int, int, int]:
    best_run = 1
    best_span = 0
    best_score = 0
    token_count = len(tokens)
    span_lengths = [2, 3, 4, 6, 8, 12, 16, 24]
    for span_len in span_lengths:
        if span_len > max_span or span_len > token_count // 2:
            continue
        idx = 0
        while idx + span_len * 2 <= token_count:
            span = tokens[idx : idx + span_len]
            if len(set(span)) == 1:
                idx += 1
                continue
            run = 1
            cursor = idx + span_len
            while cursor + span_len <= token_count and tokens[cursor : cursor + span_len] == span:
                run += 1
                cursor += span_len
            score = run * span_len
            if run >= 2 and score > best_score:
                best_run = run
                best_span = span_len
                best_score = score
            idx = cursor if run > 1 else idx + 1
    return best_run, best_span, best_score


def _ngram_repeat(tokens: list[str], n_min: int, n_max: int) -> tuple[int, int, float]:
    token_count = len(tokens)
    best_count = 1
    best_n = 0
    ngram_lengths = [2, 3, 4, 6, 8, 12]
    for ngram_len in ngram_lengths:
        if ngram_len < n_min or ngram_len > n_max or ngram_len > token_count // 2:
            continue
        counts: dict[tuple[str, ...], int] = {}
        for idx in range(token_count - ngram_len + 1):
            ngram = tuple(tokens[idx : idx + ngram_len])
            if len(set(ngram)) == 1 and ngram_len <= 3:
                continue
            if ngram in _IGNORED_SHORT_NGRAMS:
                continue
            counts[ngram] = counts.get(ngram, 0) + 1
        for count in counts.values():
            if count * ngram_len > best_count * max(best_n, 1):
                best_count = count
                best_n = ngram_len
    coverage = min(1.0, (best_count * best_n) / max(token_count, 1)) if best_n else 0.0
    return best_count, best_n, coverage


def _clause_duplication(text: str, min_normalized_chars: int) -> tuple[int, float]:
    clauses = [clause.strip() for clause in _CLAUSE_SPLIT_RE.split(text) if clause.strip()]
    normalized: list[str] = []
    for clause in clauses:
        norm = _PUNCT_SPACE_RE.sub("", clause).lower()
        if len(norm) >= min_normalized_chars:
            normalized.append(norm)
    if not normalized:
        return 1, 0.0
    counts: dict[str, int] = {}
    for clause in normalized:
        counts[clause] = counts.get(clause, 0) + 1
    max_count = max(counts.values())
    duplicate_total = sum(count for count in counts.values() if count >= 2)
    return max_count, _ratio(duplicate_total, len(normalized))


def _repetition_features(
    text: str,
    *,
    config: Mapping[str, Any],
    deep: bool = True,
) -> dict[str, int | float]:
    max_tokens = _cfg_int(config, "max_analysis_tokens", 1200)
    max_chars = _cfg_int(config, "max_analysis_chars", 12000)
    max_char_unit_len = _cfg_int(config, "max_char_unit_len", 16)
    max_immediate_span = _cfg_int(config, "max_immediate_span_tokens", 24)
    ngram_cfg = _nested_config(config, "token_ngram")
    short_ngram_cfg = _nested_config(config, "short_ngram")
    clause_cfg = _nested_config(config, "normalized_clause_duplication")

    tokens = _bounded_tokens(text, max_tokens=max_tokens)
    if deep:
        immediate_run, immediate_span, immediate_score = _immediate_span_repeat(
            tokens,
            max_span=max_immediate_span,
        )
        ngram_count, ngram_n, ngram_coverage = _ngram_repeat(
            tokens,
            n_min=_cfg_int(ngram_cfg, "n_min", 3),
            n_max=_cfg_int(ngram_cfg, "n_max", 12),
        )
        short_ngram_count, short_ngram_n, short_ngram_coverage = _ngram_repeat(
            tokens,
            n_min=_cfg_int(short_ngram_cfg, "n_min", 2),
            n_max=_cfg_int(short_ngram_cfg, "n_max", 2),
        )
    else:
        immediate_run, immediate_span, immediate_score = (1, 0, 0)
        ngram_count, ngram_n, ngram_coverage = (1, 0, 0.0)
        short_ngram_count, short_ngram_n, short_ngram_coverage = (1, 0, 0.0)
    clause_count, clause_ratio = _clause_duplication(
        text,
        min_normalized_chars=_cfg_int(clause_cfg, "min_normalized_chars", 10),
    )

    return {
        "char_repeat_run": _max_char_repeat(
            text,
            max_unit=max_char_unit_len,
            max_chars=max_chars,
        ),
        "word_repeat_run": _max_word_repeat_tokens(tokens),
        "max_token_frequency": _max_token_frequency(tokens),
        "max_token_frequency_ratio": _ratio(_max_token_frequency(tokens), len(tokens)),
        "max_ko_function_token_frequency": _max_korean_function_token_frequency(tokens),
        "max_ko_function_token_ratio": _ratio(_max_korean_function_token_frequency(tokens), len(tokens)),
        "immediate_span_run": immediate_run,
        "immediate_span_len": immediate_span,
        "immediate_span_score": immediate_score,
        "ngram_repeat_count": ngram_count,
        "ngram_n": ngram_n,
        "ngram_coverage": ngram_coverage,
        "ngram_repeated_tokens": ngram_count * ngram_n,
        "short_ngram_repeat_count": short_ngram_count,
        "short_ngram_n": short_ngram_n,
        "short_ngram_coverage": short_ngram_coverage,
        "short_ngram_repeated_tokens": short_ngram_count * short_ngram_n,
        "clause_dup_max_count": clause_count,
        "clause_dup_ratio": clause_ratio,
    }


def _repetition_flags(source: str, text: str, config: Mapping[str, Any]) -> list[str]:
    repetition_cfg = _nested_config(config, "repetition")
    if not repetition_cfg:
        repetition_cfg = config
    student = _repetition_features(text, config=repetition_cfg, deep=False)
    immediate_cfg = _nested_config(repetition_cfg, "immediate_span")
    ngram_cfg = _nested_config(repetition_cfg, "token_ngram")
    short_ngram_cfg = _nested_config(repetition_cfg, "short_ngram")
    clause_cfg = _nested_config(repetition_cfg, "normalized_clause_duplication")
    token_frequency_cfg = _nested_config(repetition_cfg, "token_frequency")
    function_token_cfg = _nested_config(repetition_cfg, "korean_function_token")
    ngram_repeat_min = _cfg_int(ngram_cfg, "repeat_count_min", 4)

    cheap_candidate = (
        student["char_repeat_run"] >= _cfg_int(repetition_cfg, "char_repeat_run_min", 16)
        or student["word_repeat_run"] >= _cfg_int(repetition_cfg, "word_repeat_run_min", 8)
        or student["max_token_frequency"] >= _cfg_int(token_frequency_cfg, "count_min", 30)
        or student["max_ko_function_token_frequency"] >= _cfg_int(function_token_cfg, "count_min", 15)
        or student["max_token_frequency"] >= _cfg_int(short_ngram_cfg, "cheap_token_frequency_min", 8)
        or (
            student["clause_dup_max_count"]
            >= _cfg_int(
                clause_cfg,
                "duplicate_count_min",
                3,
            )
            and student["clause_dup_ratio"]
            >= _cfg_float(
                clause_cfg,
                "duplicate_ratio_min",
                0.30,
            )
        )
        or (
            student["clause_dup_max_count"] >= _cfg_int(clause_cfg, "short_duplicate_count_min", 2)
            and student["clause_dup_ratio"] >= _cfg_float(clause_cfg, "short_duplicate_ratio_min", 0.80)
            and len(text) <= _cfg_int(clause_cfg, "short_text_chars_max", 1000)
        )
        or student["max_token_frequency"] >= ngram_repeat_min
    )
    if not cheap_candidate:
        return []

    student = _repetition_features(text, config=repetition_cfg, deep=True)
    abs_candidate = (
        student["char_repeat_run"] >= _cfg_int(repetition_cfg, "char_repeat_run_min", 16)
        or student["word_repeat_run"] >= _cfg_int(repetition_cfg, "word_repeat_run_min", 8)
        or (
            student["immediate_span_run"] >= _cfg_int(immediate_cfg, "repeat_run_min", 3)
            and student["immediate_span_score"] >= _cfg_int(immediate_cfg, "repeated_token_score_min", 8)
        )
        or (
            student["clause_dup_max_count"] >= _cfg_int(clause_cfg, "duplicate_count_min", 3)
            and student["clause_dup_ratio"] >= _cfg_float(clause_cfg, "duplicate_ratio_min", 0.30)
        )
        or (
            student["ngram_repeat_count"] >= ngram_repeat_min
            and student["ngram_coverage"] >= _cfg_float(ngram_cfg, "coverage_min", 0.18)
        )
        or (
            student["max_token_frequency"] >= _cfg_int(token_frequency_cfg, "count_min", 30)
            and student["max_token_frequency_ratio"] >= _cfg_float(token_frequency_cfg, "ratio_min", 0.06)
        )
        or (
            student["max_ko_function_token_frequency"] >= _cfg_int(function_token_cfg, "count_min", 15)
            and student["max_ko_function_token_ratio"] >= _cfg_float(function_token_cfg, "ratio_min", 0.10)
        )
        or (
            student["clause_dup_max_count"] >= _cfg_int(clause_cfg, "short_duplicate_count_min", 2)
            and student["clause_dup_ratio"] >= _cfg_float(clause_cfg, "short_duplicate_ratio_min", 0.80)
            and len(text) <= _cfg_int(clause_cfg, "short_text_chars_max", 1000)
        )
        or (
            student["short_ngram_repeat_count"] >= _cfg_int(short_ngram_cfg, "repeat_count_min", 8)
            and student["short_ngram_coverage"] >= _cfg_float(short_ngram_cfg, "coverage_min", 0.04)
        )
    )
    if not abs_candidate:
        return []

    source_features = _repetition_features(source, config=repetition_cfg, deep=True)
    flags: list[str] = []

    if (
        student["char_repeat_run"] >= _cfg_int(repetition_cfg, "char_repeat_run_min", 16)
        and student["char_repeat_run"]
        >= source_features["char_repeat_run"] + _cfg_int(repetition_cfg, "char_repeat_excess_over_source_min", 8)
    ):
        flags.append("repetition_char")
    if (
        student["word_repeat_run"] >= _cfg_int(repetition_cfg, "word_repeat_run_min", 8)
        and student["word_repeat_run"]
        >= source_features["word_repeat_run"] + _cfg_int(repetition_cfg, "word_repeat_excess_over_source_min", 5)
    ):
        flags.append("repetition_word")
    if (
        student["immediate_span_run"] >= _cfg_int(immediate_cfg, "repeat_run_min", 3)
        and student["immediate_span_len"] >= _cfg_int(immediate_cfg, "token_span_min", 2)
        and student["immediate_span_score"] >= _cfg_int(immediate_cfg, "repeated_token_score_min", 8)
        and student["immediate_span_score"]
        >= source_features["immediate_span_score"] + _cfg_int(immediate_cfg, "excess_score_over_source_min", 6)
    ):
        flags.append("repetition_immediate_span")
    if (
        student["immediate_span_run"] >= _cfg_int(immediate_cfg, "long_span_repeat_run_min", 2)
        and student["immediate_span_len"] >= _cfg_int(immediate_cfg, "long_span_token_span_min", 6)
        and student["immediate_span_score"] >= _cfg_int(immediate_cfg, "long_span_repeated_token_score_min", 14)
        and student["immediate_span_score"]
        >= source_features["immediate_span_score"]
        + _cfg_int(immediate_cfg, "long_span_excess_score_over_source_min", 8)
    ):
        flags.append("repetition_long_immediate_span")
    if (
        student["clause_dup_max_count"] >= _cfg_int(clause_cfg, "duplicate_count_min", 3)
        and student["clause_dup_ratio"] >= _cfg_float(clause_cfg, "duplicate_ratio_min", 0.30)
        and student["clause_dup_max_count"]
        >= source_features["clause_dup_max_count"] + _cfg_int(clause_cfg, "duplicate_count_excess_over_source_min", 2)
    ):
        flags.append("repetition_clause_dup")
    if (
        student["clause_dup_max_count"] >= _cfg_int(clause_cfg, "short_duplicate_count_min", 2)
        and student["clause_dup_ratio"] >= _cfg_float(clause_cfg, "short_duplicate_ratio_min", 0.80)
        and len(text) <= _cfg_int(clause_cfg, "short_text_chars_max", 1000)
        and student["clause_dup_max_count"]
        >= source_features["clause_dup_max_count"] + _cfg_int(clause_cfg, "short_duplicate_count_excess_over_source_min", 1)
    ):
        flags.append("repetition_short_clause_dup")
    if (
        student["ngram_repeat_count"] >= _cfg_int(ngram_cfg, "repeat_count_min", 4)
        and student["ngram_n"] >= _cfg_int(ngram_cfg, "n_min", 3)
        and student["ngram_coverage"] >= _cfg_float(ngram_cfg, "coverage_min", 0.18)
        and student["ngram_repeated_tokens"]
        >= source_features["ngram_repeated_tokens"] + _cfg_int(ngram_cfg, "repeated_token_excess_over_source_min", 8)
    ):
        flags.append("repetition_ngram")
    if (
        student["short_ngram_repeat_count"] >= _cfg_int(short_ngram_cfg, "repeat_count_min", 8)
        and student["short_ngram_n"] >= _cfg_int(short_ngram_cfg, "n_min", 2)
        and student["short_ngram_coverage"] >= _cfg_float(short_ngram_cfg, "coverage_min", 0.04)
        and student["short_ngram_repeated_tokens"]
        >= source_features["short_ngram_repeated_tokens"]
        + _cfg_int(short_ngram_cfg, "repeated_token_excess_over_source_min", 10)
    ):
        flags.append("repetition_short_ngram")
    if (
        student["max_token_frequency"] >= _cfg_int(token_frequency_cfg, "count_min", 30)
        and student["max_token_frequency_ratio"] >= _cfg_float(token_frequency_cfg, "ratio_min", 0.06)
        and student["max_token_frequency"]
        >= source_features["max_token_frequency"] + _cfg_int(token_frequency_cfg, "count_excess_over_source_min", 20)
    ):
        flags.append("repetition_token_frequency")
    if (
        student["max_ko_function_token_frequency"] >= _cfg_int(function_token_cfg, "count_min", 15)
        and student["max_ko_function_token_ratio"] >= _cfg_float(function_token_cfg, "ratio_min", 0.10)
    ):
        flags.append("repetition_ko_function_token")
    return flags


def _mixed_script_token_count(text: str) -> int:
    count = 0
    for token in _WORD_RE.findall(text):
        if len(token) >= 5 and _HANGUL_RE.search(token) and (_LATIN_RE.search(token) or _LATIN_EXT_RE.search(token)):
            count += 1
    return count


def _has_sentence_final_punctuation(text: str) -> bool:
    return bool(_SENTENCE_FINAL_RE.search(text.rstrip()))


def _source_ends_like_complete_sentence(source: str) -> bool:
    stripped = source.rstrip()
    if not _has_sentence_final_punctuation(stripped):
        return False
    if re.search(r"\b[A-Z]\.$", stripped):
        return False
    return True


def _has_dangling_korean_suffix(source: str, text: str, config: Mapping[str, Any]) -> bool:
    truncation_cfg = _nested_config(config, "truncation")
    min_chars = _cfg_int(truncation_cfg, "min_dangling_text_chars", 30)
    if len(text.strip()) < min_chars:
        return False
    if _has_sentence_final_punctuation(text):
        return False
    if not _source_ends_like_complete_sentence(source):
        return False
    suffixes = truncation_cfg.get(
        "dangling_korean_suffixes",
        [
            "이었으며",
            "였으며",
            "었으며",
            "했으며",
            "되었으며",
            "이며",
            "하며",
            "하고",
            "했고",
            "하였고",
            "되었고",
            "하기 위해",
            "위해",
            "때문에",
        ],
    )
    if not isinstance(suffixes, list):
        return False
    stripped = text.strip()
    if any(stripped.endswith(str(suffix)) for suffix in suffixes):
        return True
    particle_min_chars = _cfg_int(truncation_cfg, "min_dangling_particle_text_chars", 80)
    particles = truncation_cfg.get(
        "dangling_korean_particles",
        ["에", "의", "로", "으로", "와", "과", "은", "는", "을", "를", "이", "가", "에서", "부터", "까지"],
    )
    if len(stripped) < particle_min_chars or not isinstance(particles, list):
        return False
    return any(stripped.endswith(str(particle)) for particle in particles)


def classify_student_output(
    *,
    source: str,
    mt: str,
    status: str,
    finish_reason: Any,
    config: Mapping[str, Any],
) -> tuple[str, list[str]]:
    if status != "ok":
        return "invalid_status", ["invalid_status"]
    text = mt.strip()
    if not text:
        return "empty", ["empty"]

    flags: list[str] = []
    counts = _script_counts(text)
    text_chars = max(1, len(text.strip()))
    hangul_ratio = _ratio(counts["hangul"], text_chars)
    latin_ratio = _ratio(counts["latin"], text_chars)
    latin_like_ratio = _ratio(counts["latin"] + counts["latin_ext"], text_chars)
    wrong_script = counts["cjk"] + counts["kana"] + counts["other"]

    if _cfg_bool(config, "reject_encoding_replchar", True) and "�" in text:
        flags.append("encoding_replchar")
    if _cfg_bool(config, "reject_thinking_trace", True) and _THINKING_TRACE_RE.search(text):
        flags.append("thinking_trace")
    if _cfg_bool(config, "reject_repetition", True):
        repetition_flags = _repetition_flags(source, text, config)
        if repetition_flags:
            flags.append("repetition")
            flags.extend(repetition_flags)
    if _cfg_bool(config, "reject_foreign_passthrough", True):
        foreign_cfg = _nested_config(config, "foreign_passthrough")
        source_overlap = _token_overlap(source, text)
        low_hangul_latin = (
            hangul_ratio < _cfg_float(foreign_cfg, "low_hangul_ratio_max", 0.10)
            and latin_like_ratio > _cfg_float(foreign_cfg, "low_hangul_latin_like_ratio_min", 0.50)
        )
        source_copied_latin = (
            hangul_ratio < _cfg_float(foreign_cfg, "hangul_ratio_max", 0.45)
            and latin_like_ratio > _cfg_float(foreign_cfg, "latin_like_ratio_min", 0.30)
            and source_overlap >= _cfg_float(foreign_cfg, "source_token_overlap_min", 0.12)
        )
        if len(text) >= _cfg_int(foreign_cfg, "min_text_chars", 80) and (low_hangul_latin or source_copied_latin):
            flags.append("foreign_passthrough")
    if _cfg_bool(config, "reject_english_passthrough", True):
        english_cfg = _nested_config(config, "english_passthrough")
        source_overlap = _token_overlap(source, text)
        if (
            hangul_ratio < _cfg_float(english_cfg, "hangul_ratio_max", 0.05)
            and latin_ratio > _cfg_float(english_cfg, "latin_ratio_min", 0.50)
            and source_overlap >= _cfg_float(english_cfg, "source_token_overlap_min", 0.55)
        ):
            flags.append("english_passthrough")
        elif (
            len(text) >= _cfg_int(english_cfg, "sentence_min_text_chars", 40)
            and hangul_ratio < _cfg_float(english_cfg, "sentence_hangul_ratio_max", 0.05)
            and latin_like_ratio >= _cfg_float(english_cfg, "sentence_latin_like_ratio_min", 0.35)
            and source_overlap >= _cfg_float(english_cfg, "sentence_source_token_overlap_min", 0.55)
            and _has_english_sentence_marker(source)
            and _has_english_sentence_marker(text)
        ):
            flags.append("english_passthrough")
    if _cfg_bool(config, "reject_wrong_script_heavy", True):
        if wrong_script >= 5 or re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF\u3040-\u30FF]{4,}", text):
            flags.append("wrong_script_heavy")
    if _cfg_bool(config, "reject_wrong_script_light", True):
        light_cfg = _nested_config(config, "wrong_script_light")
        mixed_script_tokens = _mixed_script_token_count(text)
        if len(text) >= _cfg_int(light_cfg, "min_text_chars", 30) and (
            wrong_script >= _cfg_int(light_cfg, "wrong_script_char_count_min", 1)
            or mixed_script_tokens >= _cfg_int(light_cfg, "mixed_script_token_count_min", 1)
        ):
            flags.append("wrong_script_light")
    if _cfg_bool(config, "reject_truncation", True):
        if str(finish_reason).lower() in {"length", "max_tokens"}:
            flags.append("truncation")
        elif _has_dangling_korean_suffix(source, text, config):
            flags.append("truncation")
        elif len(source) >= 160 and len(text) < max(12, int(len(source) * 0.12)):
            flags.append("truncation")
    if _cfg_bool(config, "reject_length_explosion", True):
        if len(text) > max(1000, len(source) * 4):
            flags.append("length_explosion")
    if _cfg_bool(config, "reject_low_hangul_other", True):
        if len(text) >= 40 and hangul_ratio < 0.15 and wrong_script > 0:
            flags.append("low_hangul_other")

    if flags:
        return flags[0], flags
    return "clean", []
