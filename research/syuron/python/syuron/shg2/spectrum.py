from . import *
import jax.numpy as jnp
from typing import Protocol, List

# 各パラメーターの定義（例）
as_values = jnp.array([1.0, 2.0, 3.0])       # 3つの値
bs_values = jnp.array([10.0, 20.0])           # 2つの値
cs_values = jnp.array([100.0, 200.0, 300.0])   # 3つの値

# 3次元グリッドの生成（indexing='ij' を使用）
A, B, C = jnp.meshgrid(as_values, bs_values, cs_values, indexing='ij')


class SpectrumParameterSpace():
    withs_list: List[List[float]]


class Spectrum(Protocol):
    def analyze(self, params: SpectrumParameterSpace) -> List[float]:
        ...


class NCMESpectrum(Spectrum):
    def analyze(self, params: SpectrumParameterSpace) -> List[float]:
        # TODO: 内部でsolverのインスタンスを作ったりして計算を実行する
        #  GPUでできたりする？
        ...
