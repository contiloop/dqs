from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Mapping

from io_utils import write_jsonl
from runtime_logging import configure_runtime_logging


configure_runtime_logging()


def _resolve_python(env_var: str) -> str:
    value = os.environ.get(env_var, "").strip()
    if value and Path(value).exists():
        return value
    if value:
        return value
    return sys.executable


def _metricx_env(repo_env_var: str = "METRICX_REPO_DIR") -> dict[str, str]:
    env = os.environ.copy()
    repo_dir = env.get(repo_env_var, "").strip()
    if not repo_dir:
        return env

    repo_path = Path(repo_dir).expanduser()
    if not repo_path.exists():
        raise RuntimeError(f"{repo_env_var} does not exist: {repo_path}")
    if not (repo_path / "metricx24" / "predict.py").exists():
        raise RuntimeError(f"{repo_env_var} does not look like google-research/metricx: {repo_path}")

    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_path) if not existing else f"{repo_path}{os.pathsep}{existing}"
    return env


def _metricx_runner_code() -> str:
    return textwrap.dedent(
        r'''
        from __future__ import annotations

        import argparse
        import json
        import os

        import torch
        import transformers
        from metricx24 import models


        def _read_jsonl(path: str) -> list[dict]:
            rows = []
            with open(path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    if raw_line.strip():
                        rows.append(json.loads(raw_line))
            return rows


        def _input_text(row: dict, *, is_qe: bool) -> str:
            if is_qe:
                return "source: " + row["source"] + " candidate: " + row["hypothesis"]
            return (
                "source: "
                + row["source"]
                + " candidate: "
                + row["hypothesis"]
                + " reference: "
                + row["reference"]
            )


        def _tokenize_batch(tokenizer, texts: list[str], *, max_input_length: int, device):
            encoded = [
                tokenizer(
                    text,
                    max_length=max_input_length,
                    truncation=True,
                    padding=False,
                )
                for text in texts
            ]
            input_ids = []
            attention_masks = []
            for item in encoded:
                ids = list(item["input_ids"])
                mask = list(item["attention_mask"])
                if ids:
                    ids = ids[:-1]
                    mask = mask[:-1]
                if not ids:
                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                    ids = [pad_id]
                    mask = [0]
                input_ids.append(ids)
                attention_masks.append(mask)

            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            width = max(len(ids) for ids in input_ids)
            ids_tensor = torch.full(
                (len(input_ids), width),
                fill_value=pad_id,
                dtype=torch.long,
                device=device,
            )
            mask_tensor = torch.zeros(
                (len(input_ids), width),
                dtype=torch.long,
                device=device,
            )
            for row_idx, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
                length = len(ids)
                ids_tensor[row_idx, :length] = torch.tensor(ids, dtype=torch.long, device=device)
                mask_tensor[row_idx, :length] = torch.tensor(mask, dtype=torch.long, device=device)
            return ids_tensor, mask_tensor


        def main() -> None:
            parser = argparse.ArgumentParser(description="Run DQS padded MetricX inference.")
            parser.add_argument("--tokenizer", required=True)
            parser.add_argument("--model_name_or_path", required=True)
            parser.add_argument("--max_input_length", type=int, required=True)
            parser.add_argument("--batch_size", type=int, required=True)
            parser.add_argument("--input_file", required=True)
            parser.add_argument("--output_file", required=True)
            parser.add_argument("--qe", action="store_true")
            args = parser.parse_args()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
            model = models.MT5ForRegression.from_pretrained(
                args.model_name_or_path,
                torch_dtype="auto",
            )
            model.to(device)
            model.eval()

            rows = _read_jsonl(args.input_file)
            predictions = []
            batch_size = max(1, int(args.batch_size))
            with torch.inference_mode():
                for start in range(0, len(rows), batch_size):
                    chunk = rows[start : start + batch_size]
                    texts = [_input_text(row, is_qe=args.qe) for row in chunk]
                    input_ids, attention_mask = _tokenize_batch(
                        tokenizer,
                        texts,
                        max_input_length=int(args.max_input_length),
                        device=device,
                    )
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
                    batch_predictions = output.predictions.detach().float().cpu().tolist()
                    predictions.extend(float(value) for value in batch_predictions)

            dirname = os.path.dirname(args.output_file)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(args.output_file, "w", encoding="utf-8") as handle:
                for row, prediction in zip(rows, predictions):
                    out = dict(row)
                    out["prediction"] = float(prediction)
                    handle.write(json.dumps(out, ensure_ascii=False) + "\n")


        if __name__ == "__main__":
            main()
        '''
    ).strip() + "\n"


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
        runner_path = tmp / "dqs_metricx_runner.py"
        write_jsonl(input_path, payload)
        runner_path.write_text(_metricx_runner_code(), encoding="utf-8")
        cmd = [
            metricx_python,
            str(runner_path),
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
        result = subprocess.run(cmd, text=True, capture_output=True, env=_metricx_env())
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            hint = (
                "MetricX subprocess failed. Install google-research/metricx in "
                f"${python_env_var} or set {python_env_var} to that Python. "
                "If you cloned the repo instead of installing it, set METRICX_REPO_DIR."
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
