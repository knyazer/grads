import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
from brax import envs
from tqdm import tqdm

from grads.networks import Actor

jax.config.update("jax_log_compiles", True)


class Logger:
    def __init__(self, *, use_wandb=True):
        self.use_wandb = use_wandb
        if self.use_wandb:
            import wandb

            wandb.login()  # type: ignore
            wandb.init(
                name="default",
                project="ugrads",
                settings=wandb.Settings(code_dir="."),
                save_code=True,
                group="tpu",
                magic=True,
            )

    def init(self, name, **kws):
        if self.use_wandb:
            import wandb

            wandb.init(
                name=name,
                project="ugrads",
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

            wandb.finish()  # type: ignore


wandb = Logger(use_wandb=False)


def loss(agent, key=None, epoch_index=None, env_state=None, lyapunov_factor=None):
    assert epoch_index is not None
    assert key is not None
    assert env_state is not None

    key_reset, key_scan = jax.random.split(key)

    def env_step(carry, step_index: int, epoch_index: int, agent):
        env_state, key = carry
        key, key_sample = jax.random.split(key)

        mod = jnp.mod(step_index + 1, truncation_length)

        actions = eqx.filter_vmap(agent)(env_state.obs, jax.random.split(key_sample, num_envs))
        next_state = env.step(env_state, actions.transformed)
        next_state_grad = jtu.tree_map(lambda x: x - jax.lax.stop_gradient(x), next_state)
        next_state = jtu.tree_map(
            lambda no_grad, grad: no_grad + grad * lyapunov_factor,
            jax.lax.stop_gradient(next_state),
            next_state_grad,
        )

        if truncation_length is not None:
            next_state = jax.lax.cond(mod == 0.0, jax.lax.stop_gradient, lambda x: x, next_state)

        return (next_state, key), (next_state.reward, env_state.obs)

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
def training_epoch(
    agent, opt_state, key=None, epoch_index=None, lyapunov_factor=None, optimizer=None
):
    assert key is not None
    assert optimizer is not None
    assert epoch_index is not None

    agent_dyn, agent_st = eqx.partition(agent, eqx.is_array)

    def train_trunc(carry, key):
        agent_dyn, env_state, opt_state = carry
        key, key_grad = jax.random.split(key)
        agent = eqx.combine(agent_dyn, agent_st)
        (loss, (obs, world_state)), grad = loss_grad(
            agent,
            key=key_grad,
            epoch_index=epoch_index,
            env_state=env_state,
            lyapunov_factor=lyapunov_factor,
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


parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, required=False, default=0)
parser.add_argument("--n", type=int, required=False, default=1)
args = parser.parse_args()

n_hosts = args.n
lr = 1e-3
backend = "spring"
env = envs.create(env_name="ant", backend=backend)
env = envs.training.wrap(env, episode_length=1000)
env = envs.training.EvalWrapper(env)

learning_rate = lr
truncation_length = 50
num_envs = 20
episode_length = 1000
max_gradient_norm = 1.0
lyapunov_factors = [
    0.5,
    0.6,
    0.7,
    0.8,
    0.82,
    0.84,
    0.86,
    0.88,
    0.89,
    0.9,
    0.91,
    0.92,
    0.93,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
]
n = len(lyapunov_factors)
group_size = n // n_hosts
group_start = args.id * group_size
group_end = (args.id + 1) * group_size if args.id < n_hosts - 1 else n
lyapunov_factors = lyapunov_factors[group_start:group_end]
n = len(lyapunov_factors)
print(f"Running with {lyapunov_factors}")
num_epochs = 1000
wandb = Logger(use_wandb=True)
wandb.init(name="runs_{id}")


def run(env=env):
    agent = Actor(
        key=jr.PRNGKey(42),
        observation_size=env.observation_size,
        action_size=env.action_size,
    )
    optimizer = optax.chain(optax.clip(max_gradient_norm), optax.adam(learning_rate=learning_rate))
    opt_state = optimizer.init(agent.get_trainable())
    agents = []
    opt_states = []
    for i in range(n):
        agents.append(agent)
        opt_states.append(opt_state)
    agents = jax.tree.map(lambda *xs: jnp.stack(xs), *agents, is_leaf=eqx.is_array_like)
    opt_states = jax.tree.map(lambda *xs: jnp.stack(xs), *opt_states, is_leaf=eqx.is_array_like)

    local_key = jr.PRNGKey(44)

    metrics = {}
    for it in (pbar := tqdm(range(num_epochs))):
        epoch_key, local_key = jax.random.split(local_key)
        agents, opt_states, rewards = eqx.filter_vmap(
            eqx.Partial(training_epoch, optimizer=optimizer)
        )(
            agents,
            opt_states,
            jr.split(epoch_key, n),
            jnp.array([it] * n, dtype=jnp.int32),
            jnp.array(lyapunov_factors),
        )
        pbar.set_description(f"Reward: {rewards}")

        log_dict = {f"reward_{lyapunov_factors[i]}": reward for i, reward in enumerate(rewards)}
        wandb.log(log_dict)

    return agent, metrics


out = run()
