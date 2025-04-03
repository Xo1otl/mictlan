import jax.numpy as jnp
from flax.training import train_state
from typing import Tuple, Callable, List
from flax.core.scope import FrozenVariableDict
from optax import ScalarOrSchedule
from syuron import dataset


type ModelState = train_state.TrainState


type ApplyFn = Callable
type ModelParams = FrozenVariableDict
type Loss = float


type UseState = Callable[[ScalarOrSchedule, int, List[int], int], ModelState]
type TrainStep = Callable[[ModelState, dataset.Batch,
                           'LossFn'], Tuple[ModelState, Loss]]
type LossFn = Callable[[ModelParams, dataset.Batch, ApplyFn], Loss]
