from . import *
import jax.numpy as jnp
from typing import Protocol, List, NamedTuple


class SpectrumParameterSpace(NamedTuple):
    withs: List[List[float]]
    kappa_magnitude: List[float]
    T: List[float]
    wavelength: List[float]
    A0: List[complex] = []
    B0: List[complex] = []


class Spectrum(Protocol):
    def analyze(self, params: SpectrumParameterSpace) -> jnp.ndarray:
        ...


class NCMESpectrum(Spectrum):
    def analyze(self, params: SpectrumParameterSpace) -> jnp.ndarray:
        # TODO: 内部でsolverのインスタンスを作ったりして計算を実行する
        #  GPUでできたりする？
        ...
