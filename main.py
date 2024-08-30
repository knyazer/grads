import time
from typing import Callable

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu


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

    def sample(self, key: jr.PRNGKey):
        return jr.normal(key, self.mean.shape) * jnp.exp(self.log_std) + self.mean

    def entropy(self):
        return self.log_std.sum() * 0.5


class Evaluator:
    def __init__(self, eval_env, agent, num_eval_envs, episode_length):
        self._eval_walltime = 0.0
        self.eval_env = eval_env
        self.episode_length = episode_length
        self.num_eval_envs = num_eval_envs
        self._steps_per_unroll = episode_length * num_eval_envs

    @eqx.filter_jit
    def evaluate(self, key: jr.PRNGKey, agent):
        def actor_step(key: jr.PRNGKey, env_state, policy: Callable, extra_fields):
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

        def generate_unroll(
            key: jr.PRNGKey, env_state, policy: Callable, unroll_length, extra_fields
        ):
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

    def run_evaluation(
        self, key: jr.PRNGKey, agent, training_metrics, aggregate_episodes: bool = True
    ):
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


class Action(eqx.Module):
    raw: jax.Array = None
    transformed: jax.Array = None
    distr: LogNormalDistribution = None

    def __init__(self, raw, transformed, distr):
        self.raw = raw
        self.transformed = transformed
        self.distr = distr

    def postprocess(self, apply: Callable):
        return Action(raw=self.raw, transformed=apply(self.transformed), distr=self.distr)


class ValueRange(eqx.Module):
    low: jax.Array
    high: jax.Array


class MeanNetwork(eqx.Module):
    structure: list

    def __init__(self, key: jr.PRNGKey, observation_size: int, action_size: int):
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


class Actor(eqx.Module):
    mean_network: MeanNetwork
    log_std: jax.Array
    normalizer: RunningMeanStd
    constraint: Callable

    def __init__(
        self,
        key: jr.PRNGKey,
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


import optax
from brax import envs

jax.config.update("jax_log_compiles", True)


def env_step(carry, step_index: int, epoch_index: int, agent):
    env_state, key = carry
    key, key_sample = jax.random.split(key)

    mod = jnp.mod(step_index + 1, truncation_length)
    lyapunov_multiplier = lyapunov_factor

    actions = eqx.filter_vmap(agent)(env_state.obs, jax.random.split(key_sample, num_envs))
    next_state = env.step(env_state, actions.transformed)
    next_state_grad = jtu.tree_map(lambda x: x - jax.lax.stop_gradient(x), next_state)
    next_state = jtu.tree_map(
        lambda no_grad, grad: no_grad + grad * lyapunov_multiplier,
        jax.lax.stop_gradient(next_state),
        next_state_grad,
    )

    if truncation_length is not None:
        next_state = jax.lax.cond(mod == 0.0, jax.lax.stop_gradient, lambda x: x, next_state)

    return (next_state, key), (next_state.reward, env_state.obs)


def loss(agent, key=None, epoch_index=None, env_state=None):
    assert epoch_index is not None
    assert key is not None
    assert env_state is not None

    key_reset, key_scan = jax.random.split(key)

    (end_state, _), (rewards, obs) = jax.lax.scan(
        jax.tree_util.Partial(env_step, epoch_index=epoch_index, agent=agent),
        init=(env_state, key_scan),
        xs=jnp.arange(truncation_length),
        length=truncation_length,
    )

    assert rewards.shape == (truncation_length, num_envs)

    return -jnp.sum(jnp.mean(rewards, axis=1)), (obs, end_state)


loss_grad = eqx.filter_value_and_grad(loss, has_aux=True)


@eqx.filter_jit
def training_epoch(agent, opt_state, key=None, optimizer=None, epoch_index=None):
    assert key is not None
    assert optimizer is not None
    assert epoch_index is not None

    agent_dyn, agent_st = eqx.partition(agent, eqx.is_array)

    def train_trunc(carry, key):
        agent_dyn, env_state, opt_state = carry
        key, key_grad = jax.random.split(key)
        agent = eqx.combine(agent_dyn, agent_st)
        (value, (obs, world_state)), grad = loss_grad(
            agent, key=key_grad, epoch_index=epoch_index, env_state=env_state
        )
        params_update, new_opt_state = optimizer.update(grad, opt_state, agent)
        new_agent = eqx.apply_updates(agent, params_update)
        new_normalizer = new_agent.normalizer.update_batched(
            obs.reshape((-1, env.observation_size))
        )
        new_agent = eqx.tree_at(
            where=lambda s: s.normalizer,
            pytree=new_agent,
            replace_fn=lambda _: new_normalizer,
        )
        agent_dyn2, agent_st2 = eqx.partition(new_agent, eqx.is_array)
        return (agent_dyn2, world_state, new_opt_state), obs

    key_reset, key_scan = jr.split(key, 2)
    env_state = env.reset(jax.random.split(key_reset, num_envs))

    (agent_dyn, _, opt_state), _ = jax.lax.scan(
        train_trunc,
        init=(agent_dyn, env_state, opt_state),
        xs=jr.split(key_scan, episode_length // truncation_length),
        length=episode_length // truncation_length,
    )

    return eqx.combine(agent_dyn, agent_st), opt_state


for lr in [3e-3]:
    for lf in [0.88, 0.9, 0.92]:
        env = envs.create(env_name="ant", backend="spring")
        env = envs.training.wrap(env, episode_length=1000)
        env = envs.training.EvalWrapper(env)

        learning_rate = lr
        truncation_length = 100
        num_envs = 20
        episode_length = 200
        max_gradient_norm = 1.0
        lyapunov_factor = lf
        wd = 1e-6
        num_epochs = 1200

        def run(env=env):
            agent = Actor(
                key=jr.PRNGKey(42),
                observation_size=env.observation_size,
                action_size=env.action_size,
            )

            evaluator = Evaluator(
                env,
                agent,
                num_eval_envs=num_envs,
                episode_length=episode_length,
            )

            if isinstance(learning_rate, tuple):
                lr_start, lr_end = learning_rate
            else:
                lr_start = learning_rate
                lr_end = learning_rate
            schedule = optax.cosine_decay_schedule(
                init_value=lr_start, alpha=lr_end / lr_start, decay_steps=num_epochs
            )

            optimizer = optax.chain(
                optax.clip(max_gradient_norm), optax.adamw(learning_rate=schedule, weight_decay=wd)
            )
            opt_state = optimizer.init(agent.get_trainable())

            local_key = jr.PRNGKey(44)

            metrics = {}
            for it in range(num_epochs):
                epoch_key, local_key = jax.random.split(local_key)
                agent, opt_state = training_epoch(
                    agent,
                    opt_state,
                    key=epoch_key,
                    optimizer=optimizer,
                    epoch_index=jnp.array(it, dtype=jnp.int32),
                )

                if it % 10 == 3:
                    metrics = evaluator.run_evaluation(
                        agent=agent, training_metrics=metrics, key=jr.PRNGKey(42)
                    )
                    print(f'reward {metrics["eval/episode_reward"]}')

            return agent, metrics

        out = run()
