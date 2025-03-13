from typing import NamedTuple, List, Union
import jax.numpy as jnp


class Domain(NamedTuple):
    width: float
    kappa: float


type DomainStack = List[Domain]

type DomainTensor = jnp.ndarray

type DomainStackDim = Union[List[DomainStack], DomainStack, DomainTensor]
