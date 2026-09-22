import argparse
import numpy as np
from stable_baselines3 import PPO
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spider_rl.environments import ENV_VARIANTS, get_env_class


def parse_args():
    parser = argparse.ArgumentParser(description="Genera un policy map para el modelo entrenado.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="ruta al modelo PPO a evaluar",
    )
    parser.add_argument(
        "--env",
        choices=list(ENV_VARIANTS),
        default="standard",
        help="entorno base de navegación",
    )
    parser.add_argument("--output", default="policy_map.png")
    parser.add_argument("--map-limit", type=float, default=4.0)
    parser.add_argument("--map-step", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    env_cls = get_env_class(args.env)

    model = PPO.load(args.model_path)
    env = env_cls(render_mode=None)
    mapX = np.arange(-args.map_limit, args.map_limit + args.map_step / 2, args.map_step)
    mapY = np.arange(-args.map_limit, args.map_limit + args.map_step / 2, args.map_step)
    policy_map = np.zeros((len(mapY), len(mapX)), dtype=int)

    for iy, y in enumerate(mapY):
        for ix, x in enumerate(mapX):
            obs, _ = env.reset(
                seed=args.seed,
                options={"target_init_pos": np.array([x, y], dtype=np.float32)},
            )
            action, _ = model.predict(obs, deterministic=True)
            policy_map[iy, ix] = action

    plt.figure(figsize=(8, 8))
    plt.imshow(
        policy_map,
        extent=[-args.map_limit, args.map_limit, -args.map_limit, args.map_limit],
        origin="lower",
        cmap="turbo",
    )
    plt.colorbar(label="Acción PPO")
    plt.xlabel("x del target")
    plt.ylabel("y del target")
    plt.title("Policy map PPO")
    plt.savefig(args.output, dpi=150)
    plt.close()
    env.close()
    print(f"Guardado como {args.output}")


if __name__ == "__main__":
    main()
