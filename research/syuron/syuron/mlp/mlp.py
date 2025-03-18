import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from typing import Tuple, List
from optax import ScalarOrSchedule
from syuron import dataset
from .model import ModelState, ModelParams, ApplyFn, Loss


class MLP(nn.Module):
    hidden_sizes: List[int]
    output_size: int

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_sizes:
            x = nn.Dense(features=h)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.output_size)(x)
        return x


def use_state(learning_rate: ScalarOrSchedule, input_size: int, hidden_sizes: List[int], output_size: int) -> ModelState:
    model = MLP(hidden_sizes=hidden_sizes, output_size=output_size)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones([1, input_size])
    params = model.init(rng, dummy_input)
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)
    return state


def loss_fn(params: ModelParams, batch: dataset.Batch, apply_fn: ApplyFn) -> Loss:
    preds = apply_fn(params, batch.inputs)
    square_error = jnp.square(preds - batch.outputs)
    loss = jnp.mean(square_error)
    return loss  # type: ignore


def train_step(state: ModelState, batch: dataset.Batch) -> Tuple[ModelState, Loss]:
    loss, grads = jax.value_and_grad(loss_fn)(
        state.params, batch, state.apply_fn)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss
