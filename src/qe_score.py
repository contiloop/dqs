from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from runtime_logging import configure_runtime_logging, quiet_enabled


configure_runtime_logging()


def _resolve_python(env_var: str) -> str:
    value = os.environ.get(env_var, "").strip()
    if value and Path(value).exists():
        return value
    if value:
        return value
    return sys.executable


def comet_scores(
    rows: list[Mapping[str, Any]],
    *,
    model_name: str,
    batch_size: int,
    python_env_var: str,
    include_reference: bool = False,
) -> list[float]:
    comet_python = _resolve_python(python_env_var)
    payload: list[dict[str, str]] = []
    for row in rows:
        item = {"src": str(row.get("src", "")), "mt": str(row.get("mt", ""))}
        if include_reference:
            ref = str(row.get("ref", "")).strip()
            if not ref:
                raise RuntimeError("reference-based COMET requires non-empty ref for every row")
            item["ref"] = ref
        payload.append(item)

    with tempfile.TemporaryDirectory(prefix="dqs_comet_") as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "input.json"
        output_path = tmp / "output.json"
        input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        script = (
            "import json\n"
            "from comet import download_model, load_from_checkpoint\n"
            "try:\n"
            "    import torch; gpus = 1 if torch.cuda.is_available() else 0\n"
            "except Exception:\n"
            "    gpus = 0\n"
            f"data = json.loads(open({str(input_path)!r}, encoding='utf-8').read())\n"
            f"model = load_from_checkpoint(download_model({model_name!r}))\n"
            f"pred = model.predict(data, batch_size={int(batch_size)}, gpus=gpus)\n"
            "scores = pred.get('scores') if isinstance(pred, dict) else getattr(pred, 'scores', None)\n"
            f"open({str(output_path)!r}, 'w', encoding='utf-8').write(json.dumps(scores))\n"
        )
        result = subprocess.run([comet_python, "-c", script], text=True, capture_output=quiet_enabled())
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"COMET subprocess failed with code {result.returncode}: {stderr[-2000:]}")
        if not output_path.exists():
            raise RuntimeError("COMET subprocess did not write output")
        scores = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(scores, list) or len(scores) != len(rows):
        raise RuntimeError(f"COMET score count mismatch: expected={len(rows)} actual={len(scores) if isinstance(scores, list) else 'invalid'}")
    return [float(score) for score in scores]
