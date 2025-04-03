from syuron import mlp
from syuron import dataset
from typing import Tuple
import jax.numpy as jnp


def fourier_loss_fn(params: mlp.ModelParams, batch: dataset.Batch, apply_fn: mlp.ApplyFn) -> mlp.Loss:
    preds = apply_fn(params, batch.inputs)
    square_error = jnp.square(preds - batch.outputs)
    loss = jnp.mean(square_error)
    return loss  # type: ignore


def train_and_eval(ds: dataset.Dataset, epochs: int, params: mlp.OptimizableParams) -> Tuple[mlp.ModelState, mlp.Loss]:
    return mlp.train_and_eval(
        ds,
        mlp.use_state,
        mlp.train_step,
        mlp.mse_loss_fn,
        params,
        epochs
    )
