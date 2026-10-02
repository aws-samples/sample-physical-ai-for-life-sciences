from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    continues: np.ndarray
    demonstrations: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


class EpisodeReplay:
    def __init__(self, capacity_steps: int = 100_000, seed: int = 0) -> None:
        self.capacity_steps = capacity_steps
        self.episodes: deque[Episode] = deque()
        self.total_steps = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        observations: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
        continues: list[float],
        demonstrations: list[float] | None = None,
    ) -> None:
        if len(observations) != len(actions) + 1:
            raise ValueError("An episode needs one more observation than action")
        if not (len(actions) == len(rewards) == len(continues)):
            raise ValueError("Transition arrays must have the same length")
        if demonstrations is None:
            demonstrations = [0.0] * len(actions)
        if len(demonstrations) != len(actions):
            raise ValueError("Demonstration flags must align with actions")

        episode = Episode(
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            continues=np.asarray(continues, dtype=np.float32),
            demonstrations=np.asarray(demonstrations, dtype=np.float32),
        )
        self.episodes.append(episode)
        self.total_steps += episode.length

        while self.total_steps > self.capacity_steps and len(self.episodes) > 1:
            removed = self.episodes.popleft()
            self.total_steps -= removed.length

    def can_sample(self, batch_size: int, sequence_length: int) -> bool:
        eligible = sum(ep.length >= sequence_length for ep in self.episodes)
        return eligible > 0 and self.total_steps >= batch_size * sequence_length

    def sample(self, batch_size: int, sequence_length: int) -> dict[str, np.ndarray]:
        eligible = [ep for ep in self.episodes if ep.length >= sequence_length]
        if not eligible:
            raise RuntimeError("No replay episodes are long enough to sample")

        observations = []
        actions = []
        rewards = []
        continues = []
        demonstrations = []
        for _ in range(batch_size):
            episode = eligible[int(self.rng.integers(0, len(eligible)))]
            start = int(self.rng.integers(0, episode.length - sequence_length + 1))
            end = start + sequence_length
            observations.append(episode.observations[start : end + 1])
            actions.append(episode.actions[start:end])
            rewards.append(episode.rewards[start:end])
            continues.append(episode.continues[start:end])
            demonstrations.append(episode.demonstrations[start:end])

        return {
            "observations": np.stack(observations),
            "actions": np.stack(actions),
            "rewards": np.stack(rewards),
            "continues": np.stack(continues),
            "demonstrations": np.stack(demonstrations),
        }

    def state_dict(self) -> dict:
        return {
            "capacity_steps": self.capacity_steps,
            "total_steps": self.total_steps,
            "rng_state": self.rng.bit_generator.state,
            "episodes": [
                {
                    "observations": episode.observations,
                    "actions": episode.actions,
                    "rewards": episode.rewards,
                    "continues": episode.continues,
                    "demonstrations": episode.demonstrations,
                }
                for episode in self.episodes
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        self.capacity_steps = int(state["capacity_steps"])
        self.episodes = deque(
            Episode(
                observations=np.asarray(item["observations"], dtype=np.float32),
                actions=np.asarray(item["actions"], dtype=np.int64),
                rewards=np.asarray(item["rewards"], dtype=np.float32),
                continues=np.asarray(item["continues"], dtype=np.float32),
                demonstrations=np.asarray(
                    item["demonstrations"], dtype=np.float32
                ),
            )
            for item in state["episodes"]
        )
        self.total_steps = sum(episode.length for episode in self.episodes)
        if self.total_steps != int(state["total_steps"]):
            raise ValueError("Replay checkpoint step count is inconsistent")
        self.rng.bit_generator.state = state["rng_state"]
