import argparse
import datetime
import time
from typing import Callable

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
from brax import envs


class LogNormalDistribution(eqx.Module):
    """Multivariate Log Normal distribution with diagonal covariance"""

    mean: jax.Array
    log_std: jax.Array

    def __init__(self, mean, log_std):
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
        return self.log_std.sum() * 0.5  # entropy without constant factor


class Evaluator:
    """
    Evaluates agent on the environment.
    It is not jittable, since run_evaluation needs time.time()
    """

    def __init__(self, eval_env, agent, num_eval_envs, episode_length):
        self._eval_walltime = 0.0
        self.eval_env = eval_env
        self.episode_length = episode_length
        self.num_eval_envs = num_eval_envs
        self._steps_per_unroll = episode_length * num_eval_envs

    @eqx.filter_jit
    def evaluate(self, key, agent):
        def actor_step(key, env_state, policy, extra_fields):
            """Makes a single step with the provided policy in the environment."""
            keys_policy = jr.split(key, env_state.obs.shape[0])
            action = eqx.filter_vmap(policy)(env_state.obs, keys_policy)
            next_state = self.eval_env.step(env_state, action.transformed)

            return next_state, Transition(
                observation=env_state.obs,
                action=action,
                reward=next_state.reward,
                next_observation=next_state.obs,
                # extract requested additional fields
                extras={x: next_state.info[x] for x in extra_fields},
            )

        def generate_unroll(key, env_state, policy, unroll_length, extra_fields):
            """Collects trajectories of given unroll length."""

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

    def run_evaluation(self, key, agent, training_metrics, aggregate_episodes=True):
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
    """
    Action class represents a single action taken by an agent.
    Additionally stores some useful data.

    raw: action that was a direct output of an Actor-Critic model
    transformed: action that was applied on the environment
    distr: distribution from which raw action was sampled
    """

    raw: jax.Array
    transformed: jax.Array
    distr: LogNormalDistribution

    def __init__(self, raw, transformed, distr):
        self.raw = raw
        self.transformed = transformed
        self.distr = distr

    def postprocess(self, apply):
        return Action(raw=self.raw, transformed=apply(self.transformed), distr=self.distr)


class ValueRange(eqx.Module):
    low: jax.Array
    high: jax.Array


class MeanNetwork(eqx.Module):
    structure: list

    def __init__(self, key, observation_size, action_size):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.structure = [
            eqx.nn.Linear(observation_size, 64, key=key1),
            jax.nn.tanh,
            eqx.nn.Linear(64, 64, key=key2),
            jax.nn.tanh,
            eqx.nn.Linear(64, action_size, key=key4),
        ]

        # scaling down the weights of the output layer improves performance
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
    M2: jax.Array  # sum of second moments of the samples (sum of variances)
    n: jax.Array
    size: int = eqx.field(static=True)

    # we are initializing n with two so that we don't get division by zero, ever
    # this biases the running statistics, but not really that much
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
            # error if std is complex or nan
            std = eqx.error_if(
                std,
                jnp.any(jnp.isnan(std)) | jnp.any(jnp.iscomplex(std)),
                "Standard deviation should not be nan or complex",
            )

        # clip std, so that we don't get extreme values
        std = jnp.clip(std, 1e-6, 1e6)

        # clip the extreme outliers -> more stability during training.
        # by Chebyshev inequality, ~99% of values are not clipped.
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
    """A module, that outputs action distribution for a particular state."""

    mean_network: MeanNetwork
    log_std: jax.Array  # Trainable array
    normalizer: RunningMeanStd
    constraint: Callable

    def __init__(
        self,
        key,
        observation_size,
        action_size,
        initial_std=0.5,
    ):
        self.mean_network = MeanNetwork(key, observation_size, action_size)
        self.log_std = jnp.ones((action_size,)) * jnp.log(initial_std)
        self.normalizer = RunningMeanStd(observation_size)
        # tanh action constraint is applied by default
        self.constraint = lambda x: jnp.tanh(x)

    def __call__(self, x, key=None, eval=False):
        x = self.normalizer(x, eval=eval)
        distr = LogNormalDistribution(self.mean_network(x), self.log_std)
        action = distr.sample(key)
        action = self.constraint(action)
        return Action(raw=x, transformed=action, distr=distr)

    def get_trainable(self):
        """Returns the PyTree of trainable parameters."""
        return eqx.filter(self, eqx.is_inexact_array)


class Transition(eqx.Module):
    """Represents a transition between two adjacent environment states."""

    observation: jax.Array  # observation on the current state
    action: Action  # action that was taken on the current state
    reward: float  # reward, that was given as the result of the action
    next_observation: jax.Array  # next observation
    extras: dict  # any simulator-extracted hints, like end of the episode signal

    def __init__(self, observation, action, reward, next_observation, extras={}):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.extras = extras


class Logger:
    def __init__(self, *, use_wandb=False):
        self.use_wandb = use_wandb
        if self.use_wandb:
            import wandb

            wandb.login()  # type:ignore
            wandb.init(
                project="masif2",
                settings=wandb.Settings(code_dir="."),
                save_code=True,
                group="tpu",
            )

    def init(self, name, **kws):
        if self.use_wandb:
            import wandb

            wandb.init(
                name=name,
                project="masif2",
                settings=wandb.Settings(code_dir="."),
                reinit=True,
                group="tpu",
                **kws,
            )

    def log(self, data):
        if self.use_wandb:
            import wandb

            wandb.log(data)

    def finish(self):
        if self.use_wandb:
            import wandb

            wandb.finish()  # type:ignore


wandb = Logger()


def lyapunov_schedule(lyapunov_factor, epoch_index):
    return lyapunov_factor


def env_step(carry, step_index, epoch_index, zero_grad_agent=None, gradded_zero_agent=None):
    env_state, key = carry
    key, key_sample = jax.random.split(key)

    dyn_zero_grad, stat = eqx.partition(zero_grad_agent, eqx.is_array)
    dyn_gradded_zero, _ = eqx.partition(gradded_zero_agent, eqx.is_array)

    # truncate indicator
    mod = jnp.mod(step_index + 1, truncation_length)
    lyapunov_multiplier = lyapunov_schedule(lyapunov_factor, epoch_index)
    reward_multiplier = time_discount ** (mod - truncation_length)

    dyn_combined = jtu.tree_map(lambda nograd, grad: nograd + grad, dyn_zero_grad, dyn_gradded_zero)
    restored_agent = eqx.combine(dyn_combined, stat)

    actions = eqx.filter_vmap(restored_agent)(env_state.obs, jax.random.split(key_sample, num_envs))
    next_state = env.step(env_state, actions.transformed)
    # stop the gradient for the next state, mul by luapunov schedule
    next_state_grad = jtu.tree_map(lambda x: x - jax.lax.stop_gradient(x), next_state)
    next_state = jtu.tree_map(
        lambda no_grad, grad: no_grad + grad * lyapunov_multiplier,
        jax.lax.stop_gradient(next_state),
        next_state_grad,
    )

    if truncation_length is not None:
        next_state = jax.lax.cond(mod == 0.0, jax.lax.stop_gradient, lambda x: x, next_state)

    return (next_state, key), (next_state.reward * reward_multiplier, env_state.obs)


def loss(agent, key=None, epoch_index=None):
    assert epoch_index is not None
    assert key is not None
    dyn, stat = eqx.partition(agent, eqx.is_array)
    dyn_stopped = jtu.tree_map(lambda x: jax.lax.stop_gradient(x), dyn)
    zero_grad_agent = eqx.combine(dyn_stopped, stat)
    zero_dyn = jtu.tree_map(lambda d, d_stopped: d - d_stopped, dyn, dyn_stopped)
    gradded_zero_agent = eqx.combine(zero_dyn, stat)

    key_reset, key_scan = jax.random.split(key)
    env_state = env.reset(jax.random.split(key_reset, num_envs))

    _, (rewards, obs) = jax.lax.scan(
        jax.tree_util.Partial(
            env_step,
            epoch_index=epoch_index,
            zero_grad_agent=zero_grad_agent,
            gradded_zero_agent=gradded_zero_agent,
        ),
        init=(env_state, key_scan),
        xs=jnp.arange(episode_length),
        length=episode_length,
    )

    assert rewards.shape == (episode_length, num_envs)

    # loss is (inverted) sum of the mean rewards
    return -jnp.sum(jnp.mean(rewards, axis=1)), obs


loss_grad = eqx.filter_value_and_grad(loss, has_aux=True)


@eqx.filter_jit
def training_epoch(agent, opt_state, key=None, optimizer=None, epoch_index=None):
    assert key is not None
    assert optimizer is not None
    assert epoch_index is not None
    key, key_grad = jax.random.split(key)
    (value, obs), grad = loss_grad(agent, key=key_grad, epoch_index=epoch_index)
    params_update, new_opt_state = optimizer.update(grad, opt_state, agent)
    # check the learning rate
    new_agent = eqx.apply_updates(agent, params_update)
    # update normalizer
    new_normalizer = new_agent.normalizer.update_batched(obs.reshape((-1, env.observation_size)))
    new_agent = eqx.tree_at(
        where=lambda s: s.normalizer,
        pytree=new_agent,
        replace_fn=lambda _: new_normalizer,
    )
    return new_agent, new_opt_state, obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--n_hosts", type=int, default=1, help="Number of hosts for parallel execution"
    )
    parser.add_argument("--id", type=int, default=0, help="ID of the current host")
    args = parser.parse_args()

    pfn_size = args.size

    wandb = Logger(use_wandb=args.wandb)
    date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=1))).strftime("%m%d")
    for lr in [1e-4]:
        for lf in [0.9]:
            env = envs.create(env_name="ant", backend="spring")
            env = envs.training.wrap(env, episode_length=1000)
            env = envs.training.EvalWrapper(env)

            learning_rate = lr
            truncation_length = 100
            num_envs = 40
            episode_length = 400
            max_gradient_norm = 1.0
            lyapunov_factor = lf
            wd = 1e-5
            time_discount = 0.98
            num_epochs = 1200

            wandb.init(
                project="grads",
                save_code=True,
                name=f"lr={learning_rate},tr={truncation_length},lf={lyapunov_factor},td={time_discount}",
                config={
                    "learning_rate": learning_rate,
                    "truncation_length": truncation_length,
                    "num_envs": num_envs,
                    "episode_length": episode_length,
                    "max_gradient_norm": max_gradient_norm,
                    "lyapunov_factor": lyapunov_factor,
                    "weight_decay": wd,
                },
            )

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
                    optax.clip(max_gradient_norm),
                    optax.adamw(learning_rate=schedule, weight_decay=wd),
                )
                opt_state = optimizer.init(agent.get_trainable())

                local_key = jr.PRNGKey(44)

                metrics = {}
                for it in range(num_epochs):
                    # optimization
                    epoch_key, local_key = jax.random.split(local_key)
                    agent, opt_state, other = training_epoch(
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
                        wandb.log(
                            {
                                "eval/episode_reward": metrics["eval/episode_reward"],
                                "agent_log_std": agent.log_std.mean(),
                                "lyapunov_factor": lyapunov_schedule(lyapunov_factor, it),
                            }
                        )

                return agent, metrics

            out = run()
