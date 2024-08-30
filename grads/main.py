import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
from brax import envs
from tqdm import tqdm

from grads.evaluator import Evaluator
from grads.networks import Actor

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
        (loss, (obs, world_state)), grad = loss_grad(
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
        return (agent_dyn2, world_state, new_opt_state), loss

    key_reset, key_scan = jr.split(key, 2)
    env_state = env.reset(jax.random.split(key_reset, num_envs))

    (agent_dyn, _, opt_state), losses = jax.lax.scan(
        train_trunc,
        init=(agent_dyn, env_state, opt_state),
        xs=jr.split(key_scan, episode_length // truncation_length),
        length=episode_length // truncation_length,
    )

    return eqx.combine(agent_dyn, agent_st), opt_state, -losses.sum()


for lr in [3e-3]:
    for lf in [0.88, 0.9, 0.92]:
        env = envs.create(env_name="ant", backend="spring")
        env = envs.training.wrap(env, episode_length=1000)
        env = envs.training.EvalWrapper(env)

        learning_rate = lr
        truncation_length = 50
        num_envs = 20
        episode_length = 100
        max_gradient_norm = 1.0
        lyapunov_factor = lf
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
                optax.clip(max_gradient_norm), optax.adam(learning_rate=schedule)
            )
            opt_state = optimizer.init(agent.get_trainable())

            local_key = jr.PRNGKey(44)

            metrics = {}
            for it in (pbar := tqdm(range(num_epochs))):
                epoch_key, local_key = jax.random.split(local_key)
                agent, opt_state, rewards = training_epoch(
                    agent,
                    opt_state,
                    key=epoch_key,
                    optimizer=optimizer,
                    epoch_index=jnp.array(it, dtype=jnp.int32),
                )
                pbar.set_description(f"Reward: {rewards:.2f}")

            metrics = evaluator.run_evaluation(
                agent=agent, training_metrics=metrics, key=jr.PRNGKey(42)
            )

            return agent, metrics

        out = run()
