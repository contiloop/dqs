from __future__ import annotations

import math
import unittest

import torch

from cpo_objective import FullResponseCPOConfig, full_response_cpo_loss


class FullResponseCPOObjectiveTest(unittest.TestCase):
    def test_sigmoid_preference_and_chosen_nll(self) -> None:
        config = FullResponseCPOConfig(beta=0.1, cpo_alpha=1.0)
        output = full_response_cpo_loss(
            chosen_mean_logps=torch.tensor([-1.0]),
            rejected_mean_logps=torch.tensor([-2.0]),
            chosen_token_counts=torch.tensor([2]),
            rejected_token_counts=torch.tensor([2]),
            config=config,
        )
        expected_preference = -math.log(1.0 / (1.0 + math.exp(-0.2)))
        self.assertAlmostEqual(float(output.preference_loss), expected_preference, places=6)
        self.assertAlmostEqual(float(output.chosen_nll_loss), 1.0, places=6)
        self.assertAlmostEqual(float(output.loss), expected_preference + 1.0, places=6)
        self.assertAlmostEqual(float(output.margins), 2.0, places=6)

    def test_rejects_non_sigmoid_variant(self) -> None:
        with self.assertRaises(ValueError):
            FullResponseCPOConfig(loss_type="hinge")


if __name__ == "__main__":
    unittest.main()
