from typing import NamedTuple, List, Union
import jax.numpy as jnp


class Domain(NamedTuple):
    width: float
    kappa: float


type Grating = List[Domain]

# (a, b, 2) の形状を持つテンソル、aはグレーティングの数、bはドメインの数, 最後の2はそれぞれ幅とkappaのこと
type DomainTensor = jnp.ndarray

# DomainTensor の場合 b がそろってるが、それ以外の型では揃える必要がない
type GratingDim = Union[List[Grating], Grating, DomainTensor]
