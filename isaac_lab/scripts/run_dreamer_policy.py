#!/usr/bin/env python3
"""Run a downloaded SageMaker-trained Dreamer policy in Isaac Lab."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISAAC_EXTENSION = PROJECT_ROOT / "isaac_lab" / "source" / "labflow_isaac"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(ISAAC_EXTENSION))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--policy-steps", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=7)
    add_launcher_args(parser)
    parser.set_defaults(device="cuda:0")
    return parser.parse_args()


ARGS = _parse_args()

import gymnasium as gym
import torch

import labflow_isaac
from labflow_dreamer.agent import DreamerAgent
from labflow_isaac.tasks.direct.labflow import TASK_ID
from labflow_isaac.tasks.direct.labflow.labflow_env import (
    ENVIRONMENT_VERSION,
    LabFlowIsaacEnvCfg,
)


def _load_policy(observation: torch.Tensor) -> DreamerAgent:
    agent, metadata = DreamerAgent.load(
        ARGS.model_path,
        device=ARGS.device,
        seed=ARGS.seed,
    )
    env_config = metadata.get("env_config", {})
    if env_config.get("task_id") != TASK_ID:
        raise ValueError(f"Model was not trained for {TASK_ID}")
    if env_config.get("environment_version") != ENVIRONMENT_VERSION:
        raise ValueError("Model and Isaac Lab environment versions differ")
    if agent.config.observation_dim != observation.shape[-1]:
        raise ValueError("Model and Isaac Lab observation dimensions differ")
    if agent.config.action_dim != 8:
        raise ValueError("Model and Isaac Lab action dimensions differ")
    for module in (
        agent.world_model,
        agent.actor,
        agent.critic,
        agent.target_critic,
    ):
        module.eval()
    agent.reset_policy()
    return agent


def _select_actions(
    agent: DreamerAgent,
    policy_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
):
    observation, reset_mask, action_mask = policy_inputs
    return agent.act_batch(
        observation,
        action_masks=action_mask,
        reset_mask=reset_mask,
        deterministic=True,
        return_tensor=True,
    )


def _run_policy(env_cfg: LabFlowIsaacEnvCfg) -> None:
    env = gym.make(TASK_ID, cfg=env_cfg)
    try:
        observations, _ = env.reset(seed=ARGS.seed)
        observation = observations["policy"]
        agent = _load_policy(observation)
        reset_mask = torch.ones(
            ARGS.num_envs,
            dtype=torch.bool,
            device=observation.device,
        )
        for step in range(ARGS.policy_steps):
            actions = _select_actions(
                agent,
                (
                    observation,
                    reset_mask,
                    env.unwrapped.action_mask(),
                ),
            )
            observations, _, terminated, truncated, extras = env.step(actions)
            observation = observations["policy"]
            reset_mask = terminated | truncated
            if bool(extras["completed"].any()):
                event = {"event": "workflow_completed", "step": step + 1}
                print(json.dumps(event))
    finally:
        env.close()


def main() -> None:
    """Launch Isaac Lab and execute the downloaded policy."""
    env_cfg = LabFlowIsaacEnvCfg()
    env_cfg.scene.num_envs = ARGS.num_envs
    env_cfg.sim.device = ARGS.device
    try:
        with launch_simulation(env_cfg, ARGS):
            _run_policy(env_cfg)
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()
