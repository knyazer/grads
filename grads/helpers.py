from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class LogNormalDistribution(eqx.Module):
    mean: jax.Array
    log_std: jax.Array

    def __init__(self, mean: jax.Array, log_std: jax.Array):
        self.mean = mean
        self.log_std = log_std

    def get_pdf(self, value):
        value = eqx.error_if(
            value,
            value.shape != self.mean.shape,
            "Wrong shapes for the mean/value of action distr",
        )
        value = eqx.error_if(
            value,
            value.shape != self.log_std.shape,
            "Wrong shapes for the std/value of action distr",
        )

        normalized = (value - self.mean) / jnp.exp(self.log_std)
        return jax.scipy.stats.norm.logpdf(normalized).sum()

    def sample(self, key):
        return jr.normal(key, self.mean.shape) * jnp.exp(self.log_std) + self.mean

    def entropy(self):
        return self.log_std.sum() * 0.5


class Action(eqx.Module):
    raw: jax.Array
    transformed: jax.Array
    distr: LogNormalDistribution

    def __init__(self, raw, transformed, distr):
        self.raw = raw
        self.transformed = transformed
        self.distr = distr

    def postprocess(self, apply: Callable):
        return Action(raw=self.raw, transformed=apply(self.transformed), distr=self.distr)


class ValueRange(eqx.Module):
    low: jax.Array
    high: jax.Array


class RunningMeanStd(eqx.Module):
    mean: jax.Array
    M2: jax.Array
    n: jax.Array
    size: int = eqx.field(static=True)

    def __init__(self, size, mean=None, M2=None, n=jnp.int32(2)):
        self.size = size
        self.mean = (jnp.zeros(size)) if mean is None else mean
        self.M2 = (jnp.zeros(size) + 1e-6) if M2 is None else M2
        self.n = n

    def __call__(self, obs, eval=False):
        std = jnp.sqrt(self.M2 / self.n)
        if not eval:
            std = eqx.error_if(
                std,
                std.shape != obs.shape,
                "Standard deviation should have the same shape as the observation, "
                + f"std shape is {std.shape} but observation shape is {obs.shape}",
            )
            std = eqx.error_if(
                std,
                jnp.any(jnp.isnan(std)) | jnp.any(jnp.iscomplex(std)),
                "Standard deviation should not be nan or complex",
            )

        std = jnp.clip(std, 1e-6, 1e6)

        processed = jnp.clip(
            (obs - jax.lax.stop_gradient(self.mean)) / jax.lax.stop_gradient(std), -10, 10
        )

        return processed

    def update_single(self, obs):
        return self.update_batched(obs[None, :])

    def update_batched(self, obs):
        obs = eqx.error_if(
            obs,
            len(obs.shape) != 2 or obs.shape[1] != self.size,
            f"Batched observation should have the shape of (_, {self.size}),"
            + f"but got {obs.shape}",
        )

        n = self.n + obs.shape[0]

        diff_to_old_mean = obs - self.mean
        new_mean = self.mean + diff_to_old_mean.sum(axis=0) / n

        diff_to_new_mean = obs - new_mean
        var_upd = jnp.sum(diff_to_old_mean * diff_to_new_mean, axis=0)
        M2 = self.M2 + var_upd

        return RunningMeanStd(self.size, mean=new_mean, M2=M2, n=n)


class Transition(eqx.Module):
    observation: jax.Array
    action: Action
    reward: float
    next_observation: jax.Array
    extras: dict

    def __init__(self, observation, action, reward, next_observation, extras={}):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.extras = extras
