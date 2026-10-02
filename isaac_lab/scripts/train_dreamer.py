#!/usr/bin/env python3
"""Train LabFlow Dreamer against the Isaac Lab vectorized environment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISAAC_EXTENSION = (
    PROJECT_ROOT / "isaac_lab" / "source" / "labflow_isaac"
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(ISAAC_EXTENSION))


def _hyperparameters() -> dict:
    path = Path("/opt/ml/input/config/hyperparameters.json")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _default(hyperparameters: dict, name: str, fallback, converter):
    return converter(hyperparameters.get(name, fallback))


HPS = _hyperparameters()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--steps", type=int, default=_default(HPS, "steps", 20_000, int)
)
parser.add_argument(
    "--warmup-steps",
    type=int,
    default=_default(HPS, "warmup_steps", 2_000, int),
)
parser.add_argument(
    "--num-envs",
    type=int,
    default=_default(HPS, "num_envs", 64, int),
)
parser.add_argument(
    "--seed", type=int, default=_default(HPS, "seed", 7, int)
)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--sequence-length", type=int, default=16)
parser.add_argument("--updates-per-step", type=int, default=1)
parser.add_argument(
    "--checkpoint-interval-steps",
    type=int,
    default=_default(HPS, "checkpoint_interval_steps", 5_000, int),
)
parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR"))
parser.add_argument(
    "--output-data-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR")
)
parser.add_argument(
    "--checkpoint-dir", default=os.environ.get("SM_CHECKPOINT_DIR")
)
add_launcher_args(parser)
parser.set_defaults(device=str(HPS.get("device", "cuda:0")))
args = parser.parse_args()

import gymnasium as gym
import numpy as np
import torch

import labflow_isaac
from labflow_dreamer.agent import DreamerAgent, DreamerConfig
from labflow_dreamer.replay import EpisodeReplay
from labflow_isaac.tasks.direct.labflow import TASK_ID
from labflow_isaac.tasks.direct.labflow.labflow_env import (
    ENVIRONMENT_VERSION,
    LabFlowIsaacEnvCfg,
)


def _load_state(path: Path, agent: DreamerAgent, replay: EpisodeReplay) -> int:
    if not path.exists():
        return 0
    try:
        state = torch.load(path, map_location=agent.device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=agent.device)
    agent.load_training_state_dict(state["agent"])
    replay.load_state_dict(state["replay"])
    print(f"Resumed Isaac Lab training from {path}")
    return int(state["total_steps"])


def _save_state(
    path: Path,
    agent: DreamerAgent,
    replay: EpisodeReplay,
    total_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format_version": 1,
            "agent": agent.training_state_dict(),
            "replay": replay.state_dict(),
            "total_steps": total_steps,
        },
        temporary,
    )
    temporary.replace(path)


def main(env_cfg: LabFlowIsaacEnvCfg) -> None:
    """Train Dreamer in the configured Isaac Lab environment."""
    local_artifacts = PROJECT_ROOT / "artifacts" / "isaac-lab"
    managed_root = Path("/opt/ml")
    managed_training = managed_root.exists()
    model_dir = Path(
        args.model_dir
        or (
            managed_root / "model"
            if managed_training
            else local_artifacts / "model"
        )
    )
    output_dir = Path(
        args.output_data_dir
        or (
            managed_root / "output" / "data"
            if managed_training
            else local_artifacts / "output"
        )
    )
    checkpoint_dir = Path(
        args.checkpoint_dir
        or (
            managed_root / "checkpoints"
            if managed_training
            else local_artifacts / "checkpoints"
        )
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(TASK_ID, cfg=env_cfg)
    base_env = env.unwrapped
    observations, extras = env.reset(seed=args.seed)
    observation = observations["policy"]
    num_envs = observation.shape[0]

    agent = DreamerAgent(
        DreamerConfig(
            observation_dim=observation.shape[-1],
            action_dim=8,
            device=str(observation.device),
        ),
        seed=args.seed,
    )
    replay = EpisodeReplay(capacity_steps=500_000, seed=args.seed)
    training_state_path = checkpoint_dir / "training_state.pt"
    total_steps = _load_state(training_state_path, agent, replay)
    next_checkpoint = (
        total_steps // args.checkpoint_interval_steps + 1
    ) * args.checkpoint_interval_steps

    agent.reset_policy()
    reset_mask = torch.ones(num_envs, dtype=torch.bool, device=observation.device)
    episode_observations = [
        [observation[index].detach().cpu().numpy()] for index in range(num_envs)
    ]
    episode_actions = [[] for _ in range(num_envs)]
    episode_rewards = [[] for _ in range(num_envs)]
    episode_continues = [[] for _ in range(num_envs)]
    episode_demonstrations = [[] for _ in range(num_envs)]
    metrics_path = output_dir / "training.csv"
    write_header = not metrics_path.exists()

    with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "environment_steps",
                "replay_steps",
                "completed_fraction",
                "model_loss",
                "actor_loss",
                "critic_loss",
            ],
        )
        if write_header:
            writer.writeheader()

        last_losses = {
            "model_loss": float("nan"),
            "actor_loss": float("nan"),
            "critic_loss": float("nan"),
        }
        while total_steps < args.steps:
            use_teacher = total_steps < args.warmup_steps
            masks = base_env.action_mask()
            if use_teacher:
                actions = base_env.heuristic_actions()
            else:
                actions = agent.act_batch(
                    observation,
                    action_masks=masks,
                    reset_mask=reset_mask,
                    exploration=0.1,
                    return_tensor=True,
                )

            next_observations, rewards, terminated, truncated, extras = env.step(
                actions
            )
            next_observation = next_observations["policy"]
            done = terminated | truncated
            terminal_observation = extras.get(
                "terminal_observation", next_observation
            )

            action_cpu = actions.detach().cpu().numpy()
            reward_cpu = rewards.detach().cpu().numpy()
            terminated_cpu = terminated.detach().cpu().numpy()
            truncated_cpu = truncated.detach().cpu().numpy()
            next_cpu = next_observation.detach().cpu().numpy()
            terminal_cpu = terminal_observation.detach().cpu().numpy()
            for index in range(num_envs):
                finished = bool(terminated_cpu[index] or truncated_cpu[index])
                transition_observation = (
                    terminal_cpu[index] if finished else next_cpu[index]
                )
                episode_observations[index].append(transition_observation)
                episode_actions[index].append(int(action_cpu[index]))
                episode_rewards[index].append(float(reward_cpu[index]))
                episode_continues[index].append(
                    0.0 if terminated_cpu[index] else 1.0
                )
                episode_demonstrations[index].append(float(use_teacher))
                if finished:
                    replay.add(
                        episode_observations[index],
                        episode_actions[index],
                        episode_rewards[index],
                        episode_continues[index],
                        episode_demonstrations[index],
                    )
                    episode_observations[index] = [next_cpu[index]]
                    episode_actions[index] = []
                    episode_rewards[index] = []
                    episode_continues[index] = []
                    episode_demonstrations[index] = []

            total_steps += num_envs
            if replay.can_sample(args.batch_size, args.sequence_length):
                for _ in range(args.updates_per_step):
                    batch = replay.sample(args.batch_size, args.sequence_length)
                    last_losses = agent.update(batch)

            if total_steps % max(num_envs * 20, 1) < num_envs:
                completed_fraction = float(
                    extras["completed"].float().mean().item()
                )
                row = {
                    "environment_steps": total_steps,
                    "replay_steps": replay.total_steps,
                    "completed_fraction": completed_fraction,
                    "model_loss": last_losses["model_loss"],
                    "actor_loss": last_losses["actor_loss"],
                    "critic_loss": last_losses["critic_loss"],
                }
                writer.writerow(row)
                metrics_file.flush()
                print(
                    f"steps={total_steps} replay={replay.total_steps} "
                    f"completed={completed_fraction:.1%} "
                    f"model_loss={last_losses['model_loss']:.4f}"
                )

            if total_steps >= next_checkpoint:
                _save_state(
                    training_state_path, agent, replay, total_steps
                )
                while next_checkpoint <= total_steps:
                    next_checkpoint += args.checkpoint_interval_steps

            observation = next_observation
            reset_mask = done

    model_path = model_dir / "labflow_isaac_dreamer.pt"
    agent.save(
        model_path,
        env_config={
            "task_id": TASK_ID,
            "environment_version": ENVIRONMENT_VERSION,
            "num_envs": num_envs,
            "physics": "Isaac Lab JetBot",
        },
        training_steps=total_steps,
    )
    _save_state(training_state_path, agent, replay, total_steps)
    with (model_dir / "model_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "task_id": TASK_ID,
                "environment_version": ENVIRONMENT_VERSION,
                "training_steps": total_steps,
                "num_envs": num_envs,
                "observation_dim": int(observation.shape[-1]),
                "action_dim": 8,
            },
            handle,
            indent=2,
        )
    env.close()
    print(f"Saved Isaac Lab model: {model_path}")


if __name__ == "__main__":
    env_cfg = LabFlowIsaacEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    try:
        with launch_simulation(env_cfg, args):
            main(env_cfg)
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        raise
