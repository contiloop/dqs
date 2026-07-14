from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpo_objective import Setting5LossConfig, setting5_loss


class Setting5ObjectiveTest(unittest.TestCase):
    def test_paper_beta_gives_margin_two(self) -> None:
        config = Setting5LossConfig(preference_beta=0.25)
        self.assertEqual(config.target_margin, 2.0)

    def test_coefficients_are_applied_exactly_once(self) -> None:
        import torch

        config = Setting5LossConfig(lambda_sft=10.0, lambda_mpo=3.0)
        result = setting5_loss(
            chosen_completion_logps=torch.tensor([-2.0, -4.0]),
            chosen_term_logps=torch.tensor([-1.0, -2.0]),
            rejected_term_logps=torch.tensor([-2.0, -4.0]),
            config=config,
        )
        # SFT = 3.0. Margins [1, 2] against target 2 give SmoothL1 [.5, 0],
        # hence mPO=.25 and total=10*3 + 3*.25. No alpha^2 term exists.
        self.assertAlmostEqual(float(result.sft_loss), 3.0)
        self.assertAlmostEqual(float(result.mpo_loss), 0.25)
        self.assertAlmostEqual(float(result.loss), 30.75)

    def test_margin_gradient_pushes_chosen_up_and_rejected_down(self) -> None:
        import torch

        chosen_completion = torch.tensor([-1.0], requires_grad=True)
        chosen_term = torch.tensor([-3.0], requires_grad=True)
        rejected_term = torch.tensor([-3.0], requires_grad=True)
        result = setting5_loss(
            chosen_completion_logps=chosen_completion,
            chosen_term_logps=chosen_term,
            rejected_term_logps=rejected_term,
            config=Setting5LossConfig(lambda_sft=1.0, lambda_mpo=1.0),
        )
        result.loss.backward()
        # Gradient descent subtracts gradients: SFT raises the chosen-completion
        # logp, while mPO raises chosen-term and lowers rejected-term logp.
        self.assertLess(float(chosen_term.grad), 0.0)
        self.assertGreater(float(rejected_term.grad), 0.0)
        self.assertEqual(float(chosen_completion.grad), -1.0)

    def test_setting_six_is_rejected_instead_of_silently_mislabelled(self) -> None:
        with self.assertRaisesRegex(ValueError, "setting 5"):
            Setting5LossConfig(paper_setting=6)

    def test_setting_five_requires_both_terms(self) -> None:
        with self.assertRaisesRegex(ValueError, "both loss mixing coefficients"):
            Setting5LossConfig(lambda_sft=0.0)
        with self.assertRaisesRegex(ValueError, "both loss mixing coefficients"):
            Setting5LossConfig(lambda_mpo=0.0)


if __name__ == "__main__":
    unittest.main()
