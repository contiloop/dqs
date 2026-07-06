from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from progress import progress
from runtime_logging import configure_runtime_logging, quiet_enabled


configure_runtime_logging()


def _resolve_python(env_var: str) -> str:
    value = os.environ.get(env_var, "").strip()
    if value and Path(value).exists():
        return value
    if value:
        return value
    return sys.executable


def resolve_gpu_ids(*, num_gpus: int | None = None, gpu_ids: Sequence[int] | None = None) -> list[int]:
    if gpu_ids is not None:
        resolved: list[int] = []
        for idx, gpu_id in enumerate(gpu_ids):
            if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
                raise RuntimeError(f"qe.selection.gpu_ids[{idx}] must be a non-negative integer")
            if gpu_id not in resolved:
                resolved.append(gpu_id)
        if not resolved:
            raise RuntimeError("qe.selection.gpu_ids must not be empty when set")
        return resolved
    count = int(1 if num_gpus is None else num_gpus)
    if count < 0:
        raise RuntimeError("qe.selection.num_gpus must be >= 0")
    return list(range(count))


def _shard_index(
    row: Mapping[str, Any],
    *,
    fallback_idx: int,
    shard_count: int,
    total_rows: int,
    shard_strategy: str,
) -> int:
    if shard_count <= 1:
        return 0
    if shard_strategy == "row_id_hash":
        import hashlib

        row_id = str(row.get("row_id", row.get("id", fallback_idx)))
        digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % shard_count
    if shard_strategy != "order_split":
        raise RuntimeError("qe.selection.shard_strategy must be one of: order_split, row_id_hash")
    order_idx = int(row.get("order_idx", fallback_idx))
    return min((order_idx * shard_count) // max(total_rows, 1), shard_count - 1)


def _comet_payload(rows: list[Mapping[str, Any]], *, include_reference: bool) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for row in rows:
        item = {"src": str(row.get("src", "")), "mt": str(row.get("mt", ""))}
        if include_reference:
            ref = str(row.get("ref", "")).strip()
            if not ref:
                raise RuntimeError("reference-based COMET requires non-empty ref for every row")
            item["ref"] = ref
        payload.append(item)
    return payload


def _run_comet_subprocess(
    rows: list[Mapping[str, Any]],
    *,
    model_name: str,
    batch_size: int,
    python_env_var: str,
    include_reference: bool,
    gpu_id: int | None = None,
) -> list[float]:
    comet_python = _resolve_python(python_env_var)
    payload = _comet_payload(rows, include_reference=include_reference)
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
            f"pred = model.predict(data, batch_size={int(batch_size)}, gpus=gpus, num_workers=1)\n"
            "scores = pred.get('scores') if isinstance(pred, dict) else getattr(pred, 'scores', None)\n"
            f"open({str(output_path)!r}, 'w', encoding='utf-8').write(json.dumps(scores))\n"
        )
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        result = subprocess.run(
            [comet_python, "-c", script],
            text=True,
            capture_output=quiet_enabled(),
            env=env,
        )
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"COMET subprocess failed with code {result.returncode}: {stderr[-2000:]}")
        if not output_path.exists():
            raise RuntimeError("COMET subprocess did not write output")
        scores = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(scores, list) or len(scores) != len(rows):
        raise RuntimeError(
            f"COMET score count mismatch: expected={len(rows)} "
            f"actual={len(scores) if isinstance(scores, list) else 'invalid'}"
        )
    return [float(score) for score in scores]


def comet_scores(
    rows: list[Mapping[str, Any]],
    *,
    model_name: str,
    batch_size: int,
    python_env_var: str,
    include_reference: bool = False,
    num_gpus: int | None = None,
    gpu_ids: Sequence[int] | None = None,
    shard_strategy: str = "order_split",
) -> list[float]:
    if not rows:
        return []
    resolved_gpu_ids = resolve_gpu_ids(num_gpus=num_gpus, gpu_ids=gpu_ids)
    active_gpu_ids = resolved_gpu_ids if resolved_gpu_ids else [None]
    if len(active_gpu_ids) <= 1 or len(rows) <= 1:
        return _run_comet_subprocess(
            rows,
            model_name=model_name,
            batch_size=batch_size,
            python_env_var=python_env_var,
            include_reference=include_reference,
            gpu_id=active_gpu_ids[0],
        )

    shard_rows: list[list[tuple[int, Mapping[str, Any]]]] = [[] for _ in active_gpu_ids]
    for idx, row in enumerate(rows):
        shard_idx = _shard_index(
            row,
            fallback_idx=idx,
            shard_count=len(active_gpu_ids),
            total_rows=len(rows),
            shard_strategy=shard_strategy,
        )
        shard_rows[shard_idx].append((idx, row))

    progress(
        "qe comet shard start",
        rows=len(rows),
        workers=len(active_gpu_ids),
        gpus=",".join(map(str, active_gpu_ids)),
    )
    scores: list[float | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=len(active_gpu_ids)) as executor:
        futures = {}
        for shard_idx, gpu_id in enumerate(active_gpu_ids):
            indexed_rows = shard_rows[shard_idx]
            if not indexed_rows:
                continue
            futures[
                executor.submit(
                    _run_comet_subprocess,
                    [row for _, row in indexed_rows],
                    model_name=model_name,
                    batch_size=batch_size,
                    python_env_var=python_env_var,
                    include_reference=include_reference,
                    gpu_id=gpu_id,
                )
            ] = (gpu_id, [idx for idx, _ in indexed_rows])
        for future in as_completed(futures):
            gpu_id, row_indices = futures[future]
            part_scores = future.result()
            if len(part_scores) != len(row_indices):
                raise RuntimeError(
                    f"COMET shard score count mismatch on gpu={gpu_id}: "
                    f"expected={len(row_indices)} actual={len(part_scores)}"
                )
            for row_idx, score in zip(row_indices, part_scores):
                scores[row_idx] = float(score)
            progress("qe comet shard done", gpu=gpu_id, rows=len(row_indices))

    if any(score is None for score in scores):
        missing = [idx for idx, score in enumerate(scores) if score is None]
        raise RuntimeError(f"COMET shard output missing scores for row indices: {missing[:10]}")
    return [float(score) for score in scores]
