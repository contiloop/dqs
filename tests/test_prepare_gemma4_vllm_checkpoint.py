from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_gemma4_vllm_checkpoint import prepare_vllm_checkpoint  # noqa: E402


class Gemma4VllmCheckpointTest(unittest.TestCase):
    def config(self, *, hidden_size: int = 8) -> dict:
        return {
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "num_hidden_layers": 4,
                "num_kv_shared_layers": 2,
                "hidden_size": hidden_size,
                "head_dim": 4,
                "global_head_dim": 4,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": 1,
                "attention_k_eq_v": False,
                "layer_types": [
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "full_attention",
                ],
            },
        }

    def write_model(
        self,
        directory: Path,
        *,
        tensors: dict[str, torch.Tensor],
        config: dict | None = None,
    ) -> None:
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps(config or self.config()), encoding="utf-8"
        )
        (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        save_file(tensors, directory / "model.safetensors")

    def source_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {
            "language_model.model.layers.0.self_attn.q_norm.weight": torch.ones(4),
        }
        for layer in (2, 3):
            prefix = f"language_model.model.layers.{layer}.self_attn"
            tensors[f"{prefix}.k_norm.weight"] = torch.full((4,), float(layer))
            tensors[f"{prefix}.k_proj.weight"] = torch.full((4, 8), float(layer))
            tensors[f"{prefix}.v_proj.weight"] = torch.full((4, 8), float(layer))
        return tensors

    def test_builds_indexed_compat_view_without_mutating_trained_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trained = root / "trained"
            source = root / "source"
            output = root / "vllm"
            trained_tensor = {
                "language_model.model.layers.0.self_attn.q_norm.weight": torch.zeros(4)
            }
            self.write_model(trained, tensors=trained_tensor)
            self.write_model(source, tensors=self.source_tensors())

            result = prepare_vllm_checkpoint(
                trained_model=trained,
                source_sft_model=source,
                output=output,
            )

            self.assertEqual(result["supplemental_key_count"], 6)
            self.assertTrue((trained / "model.safetensors").is_file())
            self.assertFalse((trained / "model.safetensors.index.json").exists())
            index = json.loads(
                (output / "model.safetensors.index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(set(index["weight_map"].values())), 2)
            supplement = output / "model-00002-of-00002.safetensors"
            with safe_open(supplement, framework="pt", device="cpu") as handle:
                self.assertEqual(len(list(handle.keys())), 6)
                self.assertTrue(
                    torch.equal(
                        handle.get_tensor(
                            "language_model.model.layers.2.self_attn.k_norm.weight"
                        ),
                        torch.full((4,), 2.0),
                    )
                )
            base_shard = output / "model-00001-of-00002.safetensors"
            self.assertEqual(base_shard.stat().st_ino, (trained / "model.safetensors").stat().st_ino)

    def test_requires_source_k_norm_for_every_shared_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trained = root / "trained"
            source = root / "source"
            tensors = self.source_tensors()
            tensors.pop("language_model.model.layers.3.self_attn.k_norm.weight")
            self.write_model(
                trained,
                tensors={"language_model.model.layers.0.self_attn.q_norm.weight": torch.zeros(4)},
            )
            self.write_model(source, tensors=tensors)
            with self.assertRaisesRegex(ValueError, "missing_layers=\\[3\\]"):
                prepare_vllm_checkpoint(
                    trained_model=trained,
                    source_sft_model=source,
                    output=root / "vllm",
                )

    def test_accepts_k_norm_already_present_in_one_shared_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trained = root / "trained"
            source = root / "source"
            self.write_model(
                trained,
                tensors={
                    "language_model.model.layers.0.self_attn.q_norm.weight": torch.zeros(4),
                    "language_model.model.layers.2.self_attn.k_norm.weight": torch.ones(4),
                },
            )
            self.write_model(source, tensors=self.source_tensors())
            result = prepare_vllm_checkpoint(
                trained_model=trained,
                source_sft_model=source,
                output=root / "vllm",
            )
            self.assertNotIn(
                "language_model.model.layers.2.self_attn.k_norm.weight",
                result["supplemental_keys"],
            )

    def test_rejects_architecture_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trained = root / "trained"
            source = root / "source"
            self.write_model(
                trained,
                tensors={"language_model.model.layers.0.self_attn.q_norm.weight": torch.zeros(4)},
            )
            self.write_model(
                source,
                tensors=self.source_tensors(),
                config=self.config(hidden_size=16),
            )
            with self.assertRaisesRegex(ValueError, "architectures differ: hidden_size"):
                prepare_vllm_checkpoint(
                    trained_model=trained,
                    source_sft_model=source,
                    output=root / "vllm",
                )


if __name__ == "__main__":
    unittest.main()
