from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from grads.helpers import Action, LogNormalDistribution, RunningMeanStd


class MeanNetwork(eqx.Module):
    structure: list

    def __init__(self, key, observation_size: int, action_size: int):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.structure = [
            eqx.nn.Linear(observation_size, 64, key=key1),
            jax.nn.tanh,
            eqx.nn.Linear(64, 64, key=key2),
            jax.nn.tanh,
            eqx.nn.Linear(64, action_size, key=key4),
        ]

        self.structure = eqx.tree_at(
            where=lambda s: s[-1].weight,
            pytree=self.structure,
            replace_fn=lambda weight: weight * jnp.array(0.1),
        )

    def __call__(self, x):
        for operator in self.structure:
            x = operator(x)
        return x


class Actor(eqx.Module):
    mean_network: MeanNetwork
    log_std: jax.Array
    normalizer: RunningMeanStd
    constraint: Callable

    def __init__(
        self,
        key,
        observation_size: int,
        action_size: int,
        initial_std: float = 0.5,
    ):
        self.mean_network = MeanNetwork(key, observation_size, action_size)
        self.log_std = jnp.ones((action_size,)) * jnp.log(initial_std)
        self.normalizer = RunningMeanStd(observation_size)
        self.constraint = lambda x: jnp.tanh(x)

    def __call__(self, x, key=None, eval=False):
        x = self.normalizer(x, eval=eval)
        distr = LogNormalDistribution(self.mean_network(x), self.log_std)
        action = distr.sample(key)
        action = self.constraint(action)
        return Action(raw=x, transformed=action, distr=distr)

    def get_trainable(self):
        return eqx.filter(self, eqx.is_inexact_array)
