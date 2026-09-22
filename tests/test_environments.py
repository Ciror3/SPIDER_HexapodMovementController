import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from spider_rl.environments import ENV_VARIANTS
from spider_rl.environments.previous_action import SpiderEnv as PreviousStepsEnv


class StandardEnvironmentTests(unittest.TestCase):
    def test_all_variants_follow_gymnasium_api(self):
        for name, env_cls in ENV_VARIANTS.items():
            with self.subTest(name=name):
                env = env_cls(render_mode=None)
                check_env(env, skip_render_check=True)
                env.close()

    def test_seed_reproduces_reset_and_transition(self):
        for name, env_cls in ENV_VARIANTS.items():
            with self.subTest(name=name):
                first = env_cls(render_mode=None)
                second = env_cls(render_mode=None)
                try:
                    obs_a, _ = first.reset(seed=42)
                    obs_b, _ = second.reset(seed=42)
                    np.testing.assert_allclose(obs_a, obs_b)
                    step_a = first.step(3)
                    step_b = second.step(3)
                    np.testing.assert_allclose(step_a[0], step_b[0])
                    self.assertAlmostEqual(step_a[1], step_b[1])
                finally:
                    first.close()
                    second.close()

    def test_observations_remain_inside_declared_space(self):
        for name, env_cls in ENV_VARIANTS.items():
            with self.subTest(name=name):
                env = env_cls(render_mode=None, max_steps=300)
                try:
                    observation, _ = env.reset(seed=8)
                    self.assertTrue(env.observation_space.contains(observation))
                    for step in range(300):
                        observation, _, terminated, truncated, _ = env.step(
                            step % env.action_space.n
                        )
                        self.assertTrue(env.observation_space.contains(observation))
                        if terminated or truncated:
                            break
                finally:
                    env.close()


class PreviousStepsEnvironmentTests(unittest.TestCase):
    def test_previous_action_is_encoded(self):
        env = PreviousStepsEnv(render_mode=None, max_steps=2)
        try:
            observation, _ = env.reset(
                seed=3,
                options={"target_init_pos": np.array([1.0, 0.0], dtype=np.float32)},
            )
            self.assertEqual(observation.shape, (14,))
            self.assertTrue(np.all(observation[2:] == 0.0))

            observation, _, _, _, _ = env.step(3)
            self.assertEqual(observation[2:].sum(), 1.0)
            self.assertEqual(observation[2 + 3], 1.0)
        finally:
            env.close()

    def test_seed_reproduces_movement_noise(self):
        first = PreviousStepsEnv(render_mode=None)
        second = PreviousStepsEnv(render_mode=None)
        options = {"target_init_pos": np.array([1.0, 0.5], dtype=np.float32)}
        try:
            first.reset(seed=99, options=options)
            second.reset(seed=99, options=options)
            transition_a = first.step(7)
            transition_b = second.step(7)
            np.testing.assert_allclose(transition_a[0], transition_b[0])
            np.testing.assert_allclose(transition_a[4]["movement"], transition_b[4]["movement"])
        finally:
            first.close()
            second.close()


if __name__ == "__main__":
    unittest.main()
