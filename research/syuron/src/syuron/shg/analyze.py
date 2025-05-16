from .use_material import *
from .domain import *
from .ncme import *
import jax.numpy as jnp
from typing import List, NamedTuple, Union
import jax


class Params(NamedTuple):
    grating_dim: GratingDim
    T_dim: Union[List[float], float, jnp.ndarray]
    wavelength_dim: Union[List[float], float, jnp.ndarray]
    fund_power_dim: Union[List[complex], complex, jnp.ndarray]
    sh_power_dim: Union[List[complex], complex, jnp.ndarray] = 0
    mesh_density: int = 100


def to_grid(val) -> jnp.ndarray:
    if isinstance(val, jnp.ndarray):
        return val
    if isinstance(val, list):
        return jnp.array(val)
    return jnp.array([val])


def to_grating_grid(val) -> jnp.ndarray:
    if isinstance(val, jnp.ndarray):
        if val.ndim != 3 or val.shape[-1] != 2:
            raise ValueError(
                "(num_gratings, num_domains, 2) の形状を持つ配列を入力してください")
        return val
    if isinstance(val[0], list):
        return jnp.array(val)
    return jnp.array([val])


def analyze(params: Params, use_material: UseMaterial, solver_fn: NCMESolverFn) -> EffTensor:
    grating_grid = to_grating_grid(params.grating_dim)
    T_grid = to_grid(params.T_dim)
    wavelength_grid = to_grid(params.wavelength_dim)
    fund_power_grid = to_grid(params.fund_power_dim)
    sh_power_grid = to_grid(params.sh_power_dim)

    T, wavelength, fund_power, sh_power = jnp.meshgrid(
        T_grid,
        wavelength_grid,
        fund_power_grid,
        sh_power_grid,
        indexing='ij'
    )

    phase_mismatch_fn = use_material(wavelength, T)

    @jax.jit
    @jax.vmap
    def mapped_solve(grating):
        return solver_fn(NCMEParams(
            fund_power=fund_power,
            sh_power=sh_power,
            phase_mismatch_fn=phase_mismatch_fn,
            grating=grating,
            mesh_density=params.mesh_density
        ))

    results = mapped_solve(grating_grid)

    return results
