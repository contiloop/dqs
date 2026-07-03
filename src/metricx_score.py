from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from io_utils import write_jsonl


def _resolve_python(env_var: str) -> str:
    value = os.environ.get(env_var, "").strip()
    if value and Path(value).exists():
        return value
    if value:
        return value
    return sys.executable


def metricx_scores(
    rows: list[Mapping[str, Any]],
    *,
    model_name: str,
    tokenizer: str,
    max_input_length: int,
    batch_size: int,
    python_env_var: str,
    module: str = "metricx24.predict",
    include_reference: bool = True,
) -> list[float]:
    metricx_python = _resolve_python(python_env_var)
    payload: list[dict[str, str]] = []
    for row in rows:
        item = {
            "source": str(row.get("src", "")),
            "hypothesis": str(row.get("mt", "")),
            "reference": str(row.get("ref", "")) if include_reference else "",
        }
        if include_reference and not item["reference"].strip():
            raise RuntimeError("reference-based MetricX requires non-empty ref for every row")
        payload.append(item)

    with tempfile.TemporaryDirectory(prefix="dqs_metricx_") as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "input.jsonl"
        output_path = tmp / "output.jsonl"
        write_jsonl(input_path, payload)
        cmd = [
            metricx_python,
            "-m",
            module,
            "--tokenizer",
            tokenizer,
            "--model_name_or_path",
            model_name,
            "--max_input_length",
            str(int(max_input_length)),
            "--batch_size",
            str(int(batch_size)),
            "--input_file",
            str(input_path),
            "--output_file",
            str(output_path),
        ]
        if not include_reference:
            cmd.append("--qe")
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            hint = (
                "MetricX subprocess failed. Install google-research/metricx in "
                f"${python_env_var} or set {python_env_var} to that Python."
            )
            raise RuntimeError(f"{hint} code={result.returncode} stderr={stderr[-2000:]}")
        if not output_path.exists():
            raise RuntimeError("MetricX subprocess did not write output")
        predictions: list[float] = []
        with output_path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                parsed = json.loads(raw_line)
                if "prediction" not in parsed:
                    raise RuntimeError(f"MetricX output line {line_no} missing prediction")
                predictions.append(float(parsed["prediction"]))
    if len(predictions) != len(rows):
        raise RuntimeError(f"MetricX score count mismatch: expected={len(rows)} actual={len(predictions)}")
    return predictions
