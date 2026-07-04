from __future__ import annotations

import contextlib
import logging
import os
import warnings
from collections.abc import Iterator


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def quiet_enabled() -> bool:
    if _env_flag("DQS_VERBOSE", False):
        return False
    return _env_flag("DQS_QUIET", True)


def configure_runtime_logging() -> None:
    if not quiet_enabled():
        return

    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_LOGGING_COLOR", "0")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("WANDB_SILENT", "true")

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.ERROR)
    for logger_name in (
        "accelerate",
        "datasets",
        "filelock",
        "huggingface_hub",
        "lightning",
        "pytorch_lightning",
        "torch",
        "transformers",
        "unsloth",
        "unsloth_zoo",
        "urllib3",
        "vllm",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


@contextlib.contextmanager
def quiet_third_party_output() -> Iterator[None]:
    if not quiet_enabled():
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
