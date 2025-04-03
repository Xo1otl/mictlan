from typing import NamedTuple, List, Union
import jax.numpy as jnp


class Domain(NamedTuple):
    width: float
    kappa: float


type Superlattice = List[Domain]

# (a, b, 2) の形状を持つテンソル、aはスーパーラティスの数、bはドメインの数
type DomainTensor = jnp.ndarray

# DomainTensor の場合 b がそろってるが、それ以外の型では揃える必要がない
type SuperlatticeDim = Union[List[Superlattice], Superlattice, DomainTensor]
