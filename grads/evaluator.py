import time
from typing import Callable

import equinox as eqx
import equinox.internal as eqxi
import jax.numpy as jnp
import jax.random as jr
from helpers import Transition


class Evaluator:
    def __init__(self, eval_env, agent, num_eval_envs, episode_length):
        self._eval_walltime = 0.0
        self.eval_env = eval_env
        self.episode_length = episode_length
        self.num_eval_envs = num_eval_envs
        self._steps_per_unroll = episode_length * num_eval_envs

    @eqx.filter_jit
    def evaluate(self, key, agent):
        def actor_step(key, env_state, policy: Callable, extra_fields):
            keys_policy = jr.split(key, env_state.obs.shape[0])
            action = eqx.filter_vmap(policy)(env_state.obs, keys_policy)
            next_state = self.eval_env.step(env_state, action.transformed)

            return next_state, Transition(
                observation=env_state.obs,
                action=action,
                reward=next_state.reward,
                next_observation=next_state.obs,
                extras={x: next_state.info[x] for x in extra_fields},
            )

        def generate_unroll(key, env_state, policy: Callable, unroll_length, extra_fields):
            def f(carry, _):
                current_key, state = carry
                current_key, next_key = jr.split(current_key)

                next_state, transition = actor_step(
                    current_key, state, policy, extra_fields=extra_fields
                )
                return (next_key, next_state), transition

            (_, final_state), data = eqxi.scan(
                f, (key, env_state), (), length=unroll_length, kind="lax"
            )
            return final_state, data

        reset_keys = jr.split(key, self.num_eval_envs)
        eval_first_state = self.eval_env.reset(reset_keys)
        return generate_unroll(
            key,
            eval_first_state,
            agent,
            unroll_length=self.episode_length,
            extra_fields=("truncation",),
        )[0]

    def run_evaluation(self, key, agent, training_metrics, aggregate_episodes: bool = True):
        t = time.time()
        eval_state = self.evaluate(key, agent)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        for fn in [jnp.mean, jnp.std]:
            suffix = "_std" if fn == jnp.std else ""
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (fn(value) if aggregate_episodes else value)
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )
        metrics["eval/avg_episode_length"] = jnp.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

        return metrics
