from typing import List, Union
import jax.numpy as jnp

# TODO: aggregateを設定してファクトリで形状を保証する

# ドメイン数をaとして(j, 2)の形状を持つテンソル
type Grating = jnp.ndarray

# (i, j, 2) の形状を持つテンソル、iはグレーティングの数、jはドメインの数
type DomainTensor = jnp.ndarray

# DomainTensor の場合 b がそろってるが、それ以外の型では揃える必要がない
type GratingDim = Union[List[Grating], Grating, DomainTensor]
