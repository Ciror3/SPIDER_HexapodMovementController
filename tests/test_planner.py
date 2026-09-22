import unittest

import numpy as np

from spider_rl.controllers.tree_search import DiscreteTreeSearchController
from spider_rl.environments.standard import SpiderEnv


class PlannerTests(unittest.TestCase):
    def test_planner_returns_a_valid_action(self):
        env = SpiderEnv(render_mode=None)
        try:
            env.reset(
                seed=1,
                options={"target_init_pos": np.array([1.0, 0.0], dtype=np.float32)},
            )
            result = DiscreteTreeSearchController(horizon=2, beam_width=8).plan(env)
            self.assertIn(result.action, range(env.action_space.n))
            self.assertGreaterEqual(len(result.sequence), 1)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
