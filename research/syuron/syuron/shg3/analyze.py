from .use_material import *
from .solver import *
import jax.numpy as jnp
from typing import List, NamedTuple, Union
import jax


class Params(NamedTuple):
    domain_stack_dim: Union[List[DomainStack], DomainStack]
    T_dim: Union[List[float], float]
    wavelength_dim: Union[List[float], float]
    fund_power_dim: Union[List[complex], complex]
    sh_power_dim: Union[List[complex], complex] = 0
    mesh_density: int = 1000


def to_grid(val) -> jnp.ndarray:
    """Convert a single parameter value or a list of values to a jnp.array."""
    if isinstance(val, list):
        return jnp.array(val)
    return jnp.array([val])


def to_domain_stack_grid(val) -> jnp.ndarray:
    if val and not isinstance(val[0], list):
        return jnp.array([val])
    return jnp.array(val)


def analyze(params: Params, use_material: UseMaterial, solver_fn: SolverFn) -> EffTensor:
    domain_stack_grid = to_domain_stack_grid(params.domain_stack_dim)
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
    def mapped_solve(domains):
        return solver_fn(NCMEParams(
            fund_power=fund_power,
            sh_power=sh_power,
            phase_mismatch_fn=phase_mismatch_fn,
            domain_stack=domains,
            mesh_density=params.mesh_density
        ))

    results = mapped_solve(domain_stack_grid)

    return results
