from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal, kl_divergence
from torch.nn import functional as F

from labflow_dreamer.models import (
    Actor,
    Critic,
    RSSMState,
    WorldModel,
    symlog,
    symexp,
)


@dataclass
class DreamerConfig:
    observation_dim: int
    action_dim: int
    embed_dim: int = 96
    deter_dim: int = 96
    stoch_dim: int = 24
    hidden_dim: int = 128
    model_lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    kl_scale: float = 0.1
    free_nats: float = 1.0
    discount: float = 0.99
    lambda_return: float = 0.95
    imagination_horizon: int = 8
    imagination_batch: int = 64
    entropy_scale: float = 0.01
    behavior_cloning_scale: float = 4.0
    target_tau: float = 0.01
    grad_clip: float = 100.0
    device: str = "auto"


class DreamerAgent:
    """A compact Dreamer-style agent for structured observations.

    It uses an RSSM world model and trains actor/critic networks on latent
    imagined trajectories. The categorical actor uses a score-function
    estimator, which keeps this educational implementation straightforward.
    """

    def __init__(self, config: DreamerConfig, seed: int = 0) -> None:
        self.config = config
        self.device = self._select_device(config.device)
        self.rng = np.random.default_rng(seed)

        self.world_model = WorldModel(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            embed_dim=config.embed_dim,
            deter_dim=config.deter_dim,
            stoch_dim=config.stoch_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        control_dim = (
            self.world_model.rssm.feature_dim + config.observation_dim
        )
        self.actor = Actor(control_dim, config.action_dim, config.hidden_dim).to(
            self.device
        )
        self.critic = Critic(control_dim, config.hidden_dim).to(self.device)
        self.target_critic = Critic(control_dim, config.hidden_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False)

        self.model_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=config.model_lr
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )

        self._policy_state: RSSMState | None = None
        self._previous_action: Tensor | None = None
        self.updates = 0

    def reset_policy(self) -> None:
        self._policy_state = None
        self._previous_action = None

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
        deterministic: bool = False,
        exploration: float = 0.0,
    ) -> int:
        mask_batch = None if action_mask is None else action_mask[None]
        actions = self.act_batch(
            observation[None],
            action_masks=mask_batch,
            deterministic=deterministic,
            exploration=exploration,
        )
        return int(actions[0])

    @torch.no_grad()
    def act_batch(
        self,
        observations: np.ndarray | Tensor,
        action_masks: np.ndarray | Tensor | None = None,
        reset_mask: np.ndarray | Tensor | None = None,
        deterministic: bool = False,
        exploration: float = 0.0,
        return_tensor: bool = False,
    ) -> np.ndarray | Tensor:
        obs = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        if obs.ndim != 2 or obs.shape[-1] != self.config.observation_dim:
            raise ValueError(
                "observations must have shape "
                f"[batch, {self.config.observation_dim}]"
            )
        batch_size = obs.shape[0]
        rssm = self.world_model.rssm
        state_batch_size = (
            None
            if self._policy_state is None
            else self._policy_state.deter.shape[0]
        )
        if self._policy_state is None or state_batch_size != batch_size:
            self._policy_state = rssm.initial_posterior(obs, sample=False)
        else:
            if self._previous_action is None:
                raise RuntimeError("Policy action state is inconsistent")
            transitioned = rssm.posterior_step(
                self._policy_state,
                self._previous_action,
                obs,
                sample=False,
            )
            if reset_mask is not None:
                reset_tensor = torch.as_tensor(
                    reset_mask, dtype=torch.bool, device=self.device
                ).reshape(batch_size)
                if reset_tensor.any():
                    initial = rssm.initial_posterior(obs, sample=False)
                    selector = reset_tensor.unsqueeze(-1)
                    transitioned = RSSMState(
                        deter=torch.where(
                            selector, initial.deter, transitioned.deter
                        ),
                        stoch=torch.where(
                            selector, initial.stoch, transitioned.stoch
                        ),
                    )
            self._policy_state = transitioned

        control_features = torch.cat(
            [rssm.feature(self._policy_state), obs], dim=-1
        )
        logits = self.actor(control_features)
        mask_tensor: Tensor | None = None
        if action_masks is not None:
            mask_tensor = torch.as_tensor(
                action_masks, dtype=torch.bool, device=self.device
            )
            if mask_tensor.shape != logits.shape:
                raise ValueError(
                    f"action_masks must have shape {tuple(logits.shape)}"
                )
            if torch.any(~mask_tensor.any(dim=-1)):
                raise ValueError("At least one action must be valid")
            logits = logits.masked_fill(~mask_tensor, -1e9)

        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = Categorical(logits=logits).sample()

        if exploration > 0.0:
            explore_rows = self.rng.random(batch_size) < exploration
            for row in np.flatnonzero(explore_rows):
                if mask_tensor is None:
                    random_action = int(
                        self.rng.integers(0, self.config.action_dim)
                    )
                else:
                    valid = (
                        torch.nonzero(mask_tensor[row], as_tuple=False)
                        .flatten()
                        .cpu()
                        .numpy()
                    )
                    random_action = int(self.rng.choice(valid))
                actions[row] = random_action

        self._previous_action = F.one_hot(
            actions,
            self.config.action_dim,
        ).float()
        if return_tensor:
            return actions
        return actions.detach().cpu().numpy()

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        observations = self._tensor(batch["observations"], torch.float32)
        actions = self._tensor(batch["actions"], torch.long)
        rewards = self._tensor(batch["rewards"], torch.float32)
        continues = self._tensor(batch["continues"], torch.float32)
        demonstrations = self._tensor(
            batch.get(
                "demonstrations",
                np.zeros_like(batch["actions"], dtype=np.float32),
            ),
            torch.float32,
        )

        observe = self.world_model.rssm.observe(observations, actions, sample=True)
        transition_features = observe.features[:, 1:]
        predicted_observations = self.world_model.observation_head(
            observe.features
        )
        predicted_rewards = self.world_model.reward_head(
            transition_features
        ).squeeze(-1)
        predicted_continues = self.world_model.continue_head(
            transition_features
        ).squeeze(-1)

        observation_loss = F.mse_loss(
            predicted_observations, observations
        )
        reward_loss = F.mse_loss(predicted_rewards, symlog(rewards))
        continue_loss = F.binary_cross_entropy_with_logits(
            predicted_continues, continues
        )

        posterior = Normal(observe.posterior_means, observe.posterior_stds)
        prior = Normal(observe.prior_means, observe.prior_stds)
        dynamics_kl = kl_divergence(
            Normal(
                observe.posterior_means.detach(),
                observe.posterior_stds.detach(),
            ),
            prior,
        ).sum(-1)
        representation_kl = kl_divergence(
            posterior,
            Normal(
                observe.prior_means.detach(),
                observe.prior_stds.detach(),
            ),
        ).sum(-1)
        kl_value = 0.8 * dynamics_kl + 0.2 * representation_kl
        kl_loss = torch.clamp(kl_value, min=self.config.free_nats).mean()

        model_loss = (
            observation_loss
            + reward_loss
            + 0.1 * continue_loss
            + self.config.kl_scale * kl_loss
        )
        self.model_optimizer.zero_grad(set_to_none=True)
        model_loss.backward()
        model_grad_norm = nn.utils.clip_grad_norm_(
            self.world_model.parameters(), self.config.grad_clip
        )
        self.model_optimizer.step()

        start_states = self._sample_start_states(
            observe.deters.detach(), observe.stochs.detach()
        )
        imagination_actor_loss, critic_loss, imagined_return = self._behavior_losses(
            start_states
        )
        with torch.no_grad():
            behavior_observe = self.world_model.rssm.observe(
                observations, actions, sample=False
            )
        behavior_control_features = torch.cat(
            [
                behavior_observe.features[:, :-1].detach(),
                observations[:, :-1],
            ],
            dim=-1,
        )
        behavior_logits = self.actor(behavior_control_features)
        action_weights = torch.ones(
            self.config.action_dim,
            dtype=torch.float32,
            device=self.device,
        )
        # Operational events are sparse compared with grid movement. Weight
        # them more heavily so the demonstration policy learns when to act,
        # not only how to navigate.
        if self.config.action_dim >= 8:
            action_weights[4] = 6.0
            action_weights[5] = 6.0
            action_weights[6] = 3.0
        behavior_loss_per_step = F.cross_entropy(
            behavior_logits.reshape(-1, self.config.action_dim),
            actions.reshape(-1),
            weight=action_weights,
            reduction="none",
        ).reshape_as(demonstrations)
        demonstration_count = demonstrations.sum()
        behavior_cloning_loss = (
            behavior_loss_per_step * demonstrations
        ).sum() / demonstration_count.clamp(min=1.0)
        actor_loss = (
            imagination_actor_loss
            + self.config.behavior_cloning_scale * behavior_cloning_loss
        )

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.grad_clip
        )
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.grad_clip
        )
        self.critic_optimizer.step()
        self._update_target_critic()
        self.updates += 1

        return {
            "model_loss": float(model_loss.detach().cpu()),
            "observation_loss": float(observation_loss.detach().cpu()),
            "reward_loss": float(reward_loss.detach().cpu()),
            "continue_loss": float(continue_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "imagination_actor_loss": float(
                imagination_actor_loss.detach().cpu()
            ),
            "behavior_cloning_loss": float(
                behavior_cloning_loss.detach().cpu()
            ),
            "critic_loss": float(critic_loss.detach().cpu()),
            "imagined_return": float(imagined_return.detach().cpu()),
            "model_grad_norm": float(torch.as_tensor(model_grad_norm).cpu()),
            "actor_grad_norm": float(torch.as_tensor(actor_grad_norm).cpu()),
            "critic_grad_norm": float(torch.as_tensor(critic_grad_norm).cpu()),
        }

    def save(
        self,
        path: str | Path,
        env_config: dict[str, Any],
        training_steps: int,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.model_state_dict(
                env_config=env_config,
                training_steps=training_steps,
            ),
            destination,
        )

    def model_state_dict(
        self,
        env_config: dict[str, Any],
        training_steps: int,
    ) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "env_config": env_config,
            "training_steps": training_steps,
            "updates": self.updates,
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }

    def training_state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "updates": self.updates,
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_training_state_dict(self, state: dict[str, Any]) -> None:
        self.world_model.load_state_dict(state["world_model"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.model_optimizer.load_state_dict(state["model_optimizer"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.updates = int(state.get("updates", 0))
        self.rng.bit_generator.state = state["rng_state"]
        self.reset_policy()

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "auto",
        seed: int = 0,
    ) -> tuple["DreamerAgent", dict[str, Any]]:
        map_location = cls._select_device(device)
        try:
            checkpoint = torch.load(
                Path(path), map_location=map_location, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(Path(path), map_location=map_location)
        config_values = dict(checkpoint["config"])
        config_values["device"] = str(map_location)
        agent = cls(DreamerConfig(**config_values), seed=seed)
        agent.world_model.load_state_dict(checkpoint["world_model"])
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.target_critic.load_state_dict(checkpoint["target_critic"])
        agent.updates = int(checkpoint.get("updates", 0))
        metadata = {
            "env_config": checkpoint["env_config"],
            "training_steps": int(checkpoint.get("training_steps", 0)),
        }
        return agent, metadata

    def _behavior_losses(
        self, start_state: RSSMState
    ) -> tuple[Tensor, Tensor, Tensor]:
        state = start_state
        log_probabilities = []
        entropies = []
        values = []
        rewards = []
        continues = []
        next_values = []
        weights = []
        weight = torch.ones(state.deter.shape[0], device=self.device)

        for _ in range(self.config.imagination_horizon):
            features = self.world_model.rssm.feature(state).detach()
            with torch.no_grad():
                predicted_observation = self.world_model.observation_head(
                    features
                )
            control_features = torch.cat(
                [features, predicted_observation], dim=-1
            )
            logits = self.actor(control_features)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            action_one_hot = F.one_hot(
                action, self.config.action_dim
            ).float()

            log_probabilities.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            values.append(self.critic(control_features))
            weights.append(weight)

            with torch.no_grad():
                state, _, _ = self.world_model.rssm.imagine_step(
                    state, action_one_hot, sample=True
                )
                next_features = self.world_model.rssm.feature(state)
                next_observation = self.world_model.observation_head(
                    next_features
                )
                next_control_features = torch.cat(
                    [next_features, next_observation], dim=-1
                )
                imagined_reward = symexp(
                    self.world_model.reward_head(next_features).squeeze(-1)
                )
                imagined_continue = torch.sigmoid(
                    self.world_model.continue_head(next_features).squeeze(-1)
                )
                target_next_value = self.target_critic(next_control_features)

            rewards.append(imagined_reward)
            continues.append(imagined_continue)
            next_values.append(target_next_value)
            weight = (
                weight
                * self.config.discount
                * imagined_continue
            )

        returns_reversed = []
        bootstrap = next_values[-1]
        for time_index in reversed(range(self.config.imagination_horizon)):
            bootstrap = rewards[time_index] + (
                self.config.discount
                * continues[time_index]
                * (
                    (1.0 - self.config.lambda_return)
                    * next_values[time_index]
                    + self.config.lambda_return * bootstrap
                )
            )
            returns_reversed.append(bootstrap)
        returns = torch.stack(list(reversed(returns_reversed)), dim=0)

        log_probs = torch.stack(log_probabilities, dim=0)
        entropy = torch.stack(entropies, dim=0)
        value_predictions = torch.stack(values, dim=0)
        loss_weights = torch.stack(weights, dim=0).detach()
        advantage = returns - value_predictions.detach()
        advantage = (advantage - advantage.mean()) / (
            advantage.std(unbiased=False) + 1e-5
        )

        actor_loss = -(
            loss_weights
            * (
                log_probs * advantage.detach()
                + self.config.entropy_scale * entropy
            )
        ).mean()
        critic_loss = (
            loss_weights
            * F.smooth_l1_loss(
                value_predictions, returns.detach(), reduction="none"
            )
        ).mean()
        return actor_loss, critic_loss, returns.mean()

    def _sample_start_states(
        self, deters: Tensor, stochs: Tensor
    ) -> RSSMState:
        flat_deters = deters.reshape(-1, deters.shape[-1])
        flat_stochs = stochs.reshape(-1, stochs.shape[-1])
        sample_size = min(self.config.imagination_batch, flat_deters.shape[0])
        indices = torch.randperm(flat_deters.shape[0], device=self.device)[
            :sample_size
        ]
        return RSSMState(
            deter=flat_deters[indices],
            stoch=flat_stochs[indices],
        )

    def _update_target_critic(self) -> None:
        tau = self.config.target_tau
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_parameter.data.lerp_(parameter.data, tau)

    def _tensor(self, value: np.ndarray, dtype: torch.dtype) -> Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(requested)
