"""Prueba mínima de instalación sin entrenar una política."""

import sys
from pathlib import Path

# También permite ejecutarlo directamente desde un clon aún no instalado.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from spider_rl.environments import ENV_VARIANTS


def main():
    for name, env_cls in ENV_VARIANTS.items():
        env = env_cls(render_mode=None, max_steps=3)
        try:
            observation, _ = env.reset(seed=7)
            for _ in range(3):
                observation, _, terminated, truncated, _ = env.step(3)
                if terminated or truncated:
                    break
            assert env.observation_space.contains(observation)
            print(f"OK {name}: observation_shape={observation.shape}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
