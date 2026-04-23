import unittest

from edge_case_generator.rewards.components import grpo_advantages


class GRPOTests(unittest.TestCase):
    def test_relative_advantages_center_rewards(self):
        rewards = [1.0, 2.0, 3.0]
        centered = grpo_advantages(rewards, mode="centered", eps=1.0e-8)
        self.assertEqual(centered, [-1.0, 0.0, 1.0])

        normalized = grpo_advantages(rewards, mode="std_normalized", eps=1.0e-8)
        self.assertAlmostEqual(sum(normalized), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
