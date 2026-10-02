from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def symlog(value: Tensor) -> Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


def symexp(value: Tensor) -> Tensor:
    return torch.sign(value) * torch.expm1(torch.abs(value))


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        layers: int = 2,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(layers):
            modules.extend([nn.Linear(current_dim, hidden_dim), nn.ELU()])
            current_dim = hidden_dim
        modules.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*modules)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


@dataclass
class RSSMState:
    deter: Tensor
    stoch: Tensor


@dataclass
class ObserveOutput:
    features: Tensor
    deters: Tensor
    stochs: Tensor
    prior_means: Tensor
    prior_stds: Tensor
    posterior_means: Tensor
    posterior_stds: Tensor


class RSSM(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        embed_dim: int = 96,
        deter_dim: int = 96,
        stoch_dim: int = 24,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        self.encoder = MLP(observation_dim, embed_dim, hidden_dim)
        self.gru = nn.GRUCell(stoch_dim + action_dim, deter_dim)
        self.prior_net = MLP(deter_dim, 2 * stoch_dim, hidden_dim)
        self.posterior_net = MLP(
            deter_dim + embed_dim, 2 * stoch_dim, hidden_dim
        )

    @property
    def feature_dim(self) -> int:
        return self.deter_dim + self.stoch_dim

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, device=device),
        )

    def initial_posterior(self, observation: Tensor, sample: bool) -> RSSMState:
        state = self.initial(observation.shape[0], observation.device)
        embed = self.encoder(observation)
        mean, std = self._stats(self.posterior_net(torch.cat([state.deter, embed], -1)))
        stoch = self._sample(mean, std, sample)
        return RSSMState(deter=state.deter, stoch=stoch)

    def posterior_step(
        self,
        previous: RSSMState,
        previous_action: Tensor,
        observation: Tensor,
        sample: bool,
    ) -> RSSMState:
        prior, _, _ = self.imagine_step(previous, previous_action, sample=sample)
        embed = self.encoder(observation)
        mean, std = self._stats(
            self.posterior_net(torch.cat([prior.deter, embed], dim=-1))
        )
        return RSSMState(
            deter=prior.deter,
            stoch=self._sample(mean, std, sample),
        )

    def imagine_step(
        self,
        previous: RSSMState,
        action_one_hot: Tensor,
        sample: bool = True,
    ) -> tuple[RSSMState, Tensor, Tensor]:
        recurrent_input = torch.cat([previous.stoch, action_one_hot], dim=-1)
        deter = self.gru(recurrent_input, previous.deter)
        mean, std = self._stats(self.prior_net(deter))
        stoch = self._sample(mean, std, sample)
        return RSSMState(deter=deter, stoch=stoch), mean, std

    def observe(
        self,
        observations: Tensor,
        actions: Tensor,
        sample: bool = True,
    ) -> ObserveOutput:
        batch_size, sequence_plus_one, _ = observations.shape
        sequence_length = sequence_plus_one - 1
        if actions.shape[:2] != (batch_size, sequence_length):
            raise ValueError("Actions must align with observation transitions")

        embeds = self.encoder(observations.reshape(-1, self.observation_dim))
        embeds = embeds.reshape(batch_size, sequence_plus_one, self.embed_dim)
        state = self.initial(batch_size, observations.device)
        initial_mean, initial_std = self._stats(
            self.posterior_net(torch.cat([state.deter, embeds[:, 0]], dim=-1))
        )
        state = RSSMState(
            deter=state.deter,
            stoch=self._sample(initial_mean, initial_std, sample),
        )

        deters = [state.deter]
        stochs = [state.stoch]
        prior_means = []
        prior_stds = []
        posterior_means = []
        posterior_stds = []

        action_vectors = F.one_hot(actions, self.action_dim).float()
        for time_index in range(sequence_length):
            prior, prior_mean, prior_std = self.imagine_step(
                state, action_vectors[:, time_index], sample=sample
            )
            posterior_mean, posterior_std = self._stats(
                self.posterior_net(
                    torch.cat([prior.deter, embeds[:, time_index + 1]], dim=-1)
                )
            )
            state = RSSMState(
                deter=prior.deter,
                stoch=self._sample(posterior_mean, posterior_std, sample),
            )

            deters.append(state.deter)
            stochs.append(state.stoch)
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            posterior_means.append(posterior_mean)
            posterior_stds.append(posterior_std)

        deters_tensor = torch.stack(deters, dim=1)
        stochs_tensor = torch.stack(stochs, dim=1)
        return ObserveOutput(
            features=torch.cat([deters_tensor, stochs_tensor], dim=-1),
            deters=deters_tensor,
            stochs=stochs_tensor,
            prior_means=torch.stack(prior_means, dim=1),
            prior_stds=torch.stack(prior_stds, dim=1),
            posterior_means=torch.stack(posterior_means, dim=1),
            posterior_stds=torch.stack(posterior_stds, dim=1),
        )

    @staticmethod
    def feature(state: RSSMState) -> Tensor:
        return torch.cat([state.deter, state.stoch], dim=-1)

    @staticmethod
    def _sample(mean: Tensor, std: Tensor, sample: bool) -> Tensor:
        if not sample:
            return mean
        return mean + std * torch.randn_like(mean)

    def _stats(self, raw_stats: Tensor) -> tuple[Tensor, Tensor]:
        mean, raw_std = torch.chunk(raw_stats, 2, dim=-1)
        std = F.softplus(raw_std) + 0.1
        return mean, std


class WorldModel(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        embed_dim: int = 96,
        deter_dim: int = 96,
        stoch_dim: int = 24,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.rssm = RSSM(
            observation_dim=observation_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            hidden_dim=hidden_dim,
        )
        feature_dim = self.rssm.feature_dim
        self.observation_head = MLP(
            feature_dim, observation_dim, hidden_dim
        )
        self.reward_head = MLP(feature_dim, 1, hidden_dim)
        self.continue_head = MLP(feature_dim, 1, hidden_dim)


class Actor(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.policy = MLP(feature_dim, action_dim, hidden_dim)

    def forward(self, features: Tensor) -> Tensor:
        return self.policy(features)


class Critic(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.value = MLP(feature_dim, 1, hidden_dim)

    def forward(self, features: Tensor) -> Tensor:
        return self.value(features).squeeze(-1)
