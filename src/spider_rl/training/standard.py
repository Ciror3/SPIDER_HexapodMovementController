import argparse
import os
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from spider_rl.environments import ENV_VARIANTS, get_env_class
from spider_rl.training.callbacks import PolicyMapCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena PPO para Spider con entornos configurables.")
    parser.add_argument(
        "--env",
        choices=list(ENV_VARIANTS),
        default="sin_obstaculos",
        help="elige el entorno: sin_obstaculos, obstaculos_sin_lidar o obstaculos_lidar",
    )
    parser.add_argument("-n", "--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--policy-map-freq", type=int, default=5_000)
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args()


def make_env(env_cls, seed: int = 0, **env_kwargs):
    def _init():
        env = env_cls(**env_kwargs)
        env = Monitor(env)  # Mantenemos el monitor para ver rewards
        env.reset(seed=seed)
        return env
    return _init


def main():
    args = parse_args()
    if args.n_envs < 1:
        raise ValueError("--n-envs debe ser mayor o igual a 1")
    if args.total_timesteps < 1:
        raise ValueError("--total-timesteps debe ser mayor o igual a 1")
    env_cls = get_env_class(args.env)

    # ---------- CONFIGURACIÓN ----------
    run_name = args.run_name or datetime.now().strftime(f"{args.env}_%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, "logs", run_name)
    models_dir = os.path.join(args.output_dir, "models", run_name)
    policy_maps_dir = os.path.join(args.output_dir, "policy_maps", run_name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(policy_maps_dir, exist_ok=True)

    # ---------- ENTORNOS EN PARALELO ----------
    env_kwargs = dict(render_mode=None)
    env_fns = [make_env(env_cls, seed=args.seed + i, **env_kwargs) for i in range(args.n_envs)]
    env = DummyVecEnv(env_fns) if args.n_envs == 1 else SubprocVecEnv(env_fns)

    # ---------- MODELO PPO ----------
    # Stable-Baselines3 adapta la red a la observación de cada variante.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.000355207825977822,
        n_steps=256,
        batch_size=8,
        n_epochs=20,
        gamma=0.9791381032304535,
        gae_lambda=0.9975914694455162,
        clip_range=0.1,
        ent_coef=1.440070886288556e-06,
        vf_coef=0.5,
        max_grad_norm=0.7456096284142648,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=torch.nn.Tanh,
        ),
        verbose=0,
        tensorboard_log=log_dir,
        seed=args.seed,
        device=args.device,
    )

    # ---------- CALLBACKS ----------
    policy_cb = PolicyMapCallback(
        freq=args.policy_map_freq,
        save_path=policy_maps_dir,
        verbose=1,
        env_cls=env_cls,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=models_dir,
        name_prefix=f"ppo_spider_{args.env}",
    )

    callbacks = [policy_cb, checkpoint_cb]

    # ---------- ENTRENAMIENTO ----------
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=not args.no_progress_bar,
        )

        # ---------- GUARDAR MODELO FINAL ----------
        final_model_path = os.path.join(models_dir, f"ppo_spider_final_{args.env}")
        model.save(final_model_path)
        print(f"Modelo final guardado en: {final_model_path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
