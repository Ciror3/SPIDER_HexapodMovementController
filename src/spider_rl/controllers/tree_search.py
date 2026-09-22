import argparse
import os
import time
from dataclasses import dataclass

import numpy as np


from spider_rl.environments import ENV_VARIANTS, get_env_class


ACTION_NAMES = {
    0: "Pivot_Left",
    1: "Pivot_Right",
    2: "FwdSteer_Left",
    3: "Fwd",
    4: "FwdSteer_Right",
    5: "BwdSteer_Left",
    6: "BwdSteer_Right",
    7: "FastFwd",
    8: "Bwd",
    9: "FastBwd",
    10: "FastFwdSteer_Left",
    11: "FastFwdSteer_Right",
}


@dataclass
class PlanningState:
    target_pos: np.ndarray
    previous_motion: str | None = None


@dataclass
class PlanResult:
    action: int
    score: float
    sequence: tuple[int, ...]
    predicted_target: np.ndarray


class DiscreteTreeSearchController:
    """Tree-search baseline over the same discrete commands used by PPO."""

    def __init__(
        self,
        horizon=3,
        beam_width=None,
        stop_on_success_depth=True,
    ):
        self.horizon = int(horizon)
        self.beam_width = None if beam_width is None else int(beam_width)
        self.stop_on_success_depth = bool(stop_on_success_depth)

    def plan(self, env):
        actions = tuple(range(env.action_space.n))
        initial = self._read_state(env)
        initial_distance = self._distance(initial)

        frontier = [(0.0, (), initial, initial_distance)]
        finished = []

        for _ in range(self.horizon):
            expanded = []
            found_success_at_depth = False

            for score, seq, state, last_distance in frontier:
                if self._is_terminal(env, state):
                    finished.append((score, seq, state, last_distance))
                    found_success_at_depth = found_success_at_depth or self._is_success(env, state)
                    continue

                for action in actions:
                    child = self._expand(env, score, seq, state, last_distance, action)
                    _, _, child_state, _ = child
                    if self._is_terminal(env, child_state):
                        finished.append(child)
                        found_success_at_depth = (
                            found_success_at_depth
                            or self._is_success(env, child_state)
                        )
                    else:
                        expanded.append(child)

            if found_success_at_depth and self.stop_on_success_depth:
                break

            if self.beam_width is not None:
                expanded.sort(key=lambda item: self._terminal_score(env, *item), reverse=True)
                expanded = expanded[: self.beam_width]

            frontier = expanded
            if not frontier:
                break

        candidates = finished + frontier
        successful = [
            item for item in candidates if self._is_success(env, item[2])
        ]
        selectable = successful if successful else candidates
        best = max(selectable, key=lambda item: self._terminal_score(env, *item))
        final_score = self._terminal_score(env, *best)
        _, sequence, state, _ = best
        return PlanResult(
            action=sequence[0] if sequence else -1,
            score=final_score,
            sequence=sequence,
            predicted_target=state.target_pos.copy(),
        )

    def _expand(self, env, score, seq, state, last_distance, action):
        next_state = self._simulate_action(env, state, action)
        dist = self._distance(next_state)
        reward = self._reward_like_env(env, state, next_state)

        return score + reward, seq + (action,), next_state, dist

    def _terminal_score(self, env, score, seq, state, last_distance):
        return score

    def _reward_like_env(self, env, state, next_state):
        last_distance = self._distance(state)
        dist = self._distance(next_state)
        terminated = dist <= env.success_radius

        new_angle = self._angle_misalignment(next_state.target_pos)
        last_angle = self._angle_misalignment(state.target_pos)
        ori_improvement = last_angle - new_angle
        ori_improvement /= np.pi / 2.0

        reward = 0.0
        distance_delta = last_distance - dist
        reward += distance_delta * env.distance_scale
        reward -= env.step_cost

        if distance_delta > 0:
            far_scale = min(1.0, dist / (env.world_size / 2.0)) if env.world_size > 0 else 0.0
            reward += env.orientation_weight * ori_improvement * far_scale

        if distance_delta < 0:
            reward -= env.backtrack_penalty

        if terminated:
            reward += env.success_bonus

        return reward

    def _is_terminal(self, env, state):
        return self._is_success(env, state)

    def _is_success(self, env, state):
        return self._distance(state) <= env.success_radius

    def _read_state(self, env):
        return PlanningState(
            target_pos=np.asarray(env.target_pos, dtype=np.float64).copy(),
            previous_motion=getattr(env, "previous_motion_name", None),
        )

    def _simulate_action(self, env, state, action):
        _, movement = self._movement_for_action(env, state, action)
        dx, dy, dtheta = np.asarray(movement, dtype=np.float64)
        target = self._transform_points(state.target_pos.reshape(1, 2), dtheta, dx, dy)[0]

        return PlanningState(
            target_pos=target,
            previous_motion=self._next_previous_motion(env, action, state.previous_motion),
        )

    @staticmethod
    def _movement_for_action(env, state, action):
        commands = env.commands
        if action in commands:
            return commands[action]

        previous_key = state.previous_motion or "__initial__"
        return commands[previous_key][action]

    @staticmethod
    def _next_previous_motion(env, action, fallback):
        motion_to_action = getattr(env, "motion_to_action", None)
        if motion_to_action is None:
            return fallback
        return ACTION_NAMES.get(action, fallback)

    @staticmethod
    def _transform_points(points, dtheta, dx, dy):
        rotation = np.array(
            [
                [np.cos(-dtheta), -np.sin(-dtheta)],
                [np.sin(-dtheta), np.cos(-dtheta)],
            ],
            dtype=np.float64,
        )
        points = np.asarray(points, dtype=np.float64)
        return (rotation @ (points - np.array([dx, dy], dtype=np.float64)).T).T

    @staticmethod
    def _distance(state):
        return float(np.linalg.norm(state.target_pos))

    @staticmethod
    def _angle_misalignment(target_pos):
        dist = float(np.linalg.norm(target_pos))
        if dist < 1e-6:
            return 0.0
        angle = float(np.arctan2(target_pos[1], target_pos[0]))
        abs_angle = abs(angle)
        return min(abs_angle, abs(np.pi - abs_angle))

def evaluate_controller(env_cls, args):
    controller = DiscreteTreeSearchController(
        horizon=args.horizon,
        beam_width=args.beam_width,
        stop_on_success_depth=not args.continue_after_first_success,
    )

    successes = 0
    returns = []
    steps = []
    wall_times = []

    total_start_time = time.perf_counter()

    for episode in range(args.episodes):
        env = env_cls(render_mode=args.render_mode, max_steps=args.max_steps)
        episode_start_time = time.perf_counter()
        np.random.seed(args.seed + episode)
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        episode_return = 0.0
        episode_actions = []

        while not done:
            plan = controller.plan(env)
            if not plan.sequence:
                break

            for action in plan.sequence:
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_actions.append(action)
                done = terminated or truncated
                if done or args.replan_each_step:
                    break

        distance = float(np.linalg.norm(env.target_pos))
        success = distance <= env.success_radius
        successes += int(success)
        returns.append(episode_return)
        steps.append(env.step_count)
        wall_times.append(time.perf_counter() - episode_start_time)

        should_print_progress = (
            args.progress_every > 0
            and ((episode + 1) % args.progress_every == 0 or episode + 1 == args.episodes)
        )
        if should_print_progress:
            running_avg_reward = float(np.mean(returns))
            running_avg_steps = float(np.mean(steps))
            print(
                f"episode={episode + 1}/{args.episodes} "
                f"success={int(success)} "
                f"steps={env.step_count} reward={episode_return:.3f} "
                f"avg_steps={running_avg_steps:.2f} avg_reward={running_avg_reward:.3f}"
            )

        if args.verbose:
            names = [ACTION_NAMES.get(action, str(action)) for action in episode_actions]
            print(
                f"episode={episode + 1} success={success} "
                f"steps={env.step_count} return={episode_return:.3f} "
                f"final_distance={distance:.3f} actions={names}"
            )

        env.close()

    n = max(args.episodes, 1)
    total_wall_time = time.perf_counter() - total_start_time
    return {
        "episodes": args.episodes,
        "success_rate": successes / n,
        "avg_reward": float(np.mean(returns)) if returns else 0.0,
        "std_reward": float(np.std(returns)) if returns else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "std_steps": float(np.std(steps)) if steps else 0.0,
        "avg_wall_time_sec": float(np.mean(wall_times)) if wall_times else 0.0,
        "std_wall_time_sec": float(np.std(wall_times)) if wall_times else 0.0,
        "total_wall_time_sec": float(total_wall_time),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline de busqueda en arbol para SpiderEnv.")
    parser.add_argument("--env", choices=list(ENV_VARIANTS), default="standard")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument(
        "--beam-width",
        type=int,
        default=128,
        help="cantidad de nodos que conserva por profundidad; más bajo es más rápido",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="expande todo el árbol completo; puede ser muy lento",
    )
    parser.add_argument(
        "--continue-after-first-success",
        action="store_true",
        help="sigue expandiendo hasta horizon aunque ya exista una ruta exitosa",
    )
    parser.add_argument(
        "--replan-each-step",
        action="store_true",
        help="ejecuta solo el primer comando del camino y vuelve a planificar",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="imprime progreso cada N episodios; usar 0 para desactivar",
    )
    parser.add_argument("--render-mode", choices=["human", "none"], default="none")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.render_mode == "none":
        args.render_mode = None
    if args.exhaustive:
        args.beam_width = None

    env_cls = get_env_class(args.env)
    metrics = evaluate_controller(env_cls, args)

    print("Tree-search baseline final")
    print(f"env={args.env} episodes={args.episodes} horizon={args.horizon} beam_width={args.beam_width}")
    print(f"avg_steps={metrics['avg_steps']:.2f} +- {metrics['std_steps']:.2f}")
    print(f"avg_wall_time_sec={metrics['avg_wall_time_sec']:.4f} +- {metrics['std_wall_time_sec']:.4f}")
    print(f"avg_reward={metrics['avg_reward']:.4f} +- {metrics['std_reward']:.4f}")
    print(f"success_rate={metrics['success_rate']:.4f}")


if __name__ == "__main__":
    main()
