from . import *
import jax.numpy as jnp
from typing import List, NamedTuple, TypeAlias, Union


class SpectrumParams(NamedTuple):
    widths_dim: Union[List[List[float]], List[float]]
    kappa_magnitude_dim: Union[List[float], float]
    T_dim: Union[List[float], float]
    wavelength_dim: Union[List[float], float]
    A0_dim: Union[List[complex], complex]
    B0_dim: Union[List[complex], complex] = 0


Spectrum: TypeAlias = jnp.ndarray


def to_param_array(val) -> jnp.ndarray:
    """Convert a single parameter value or a list of values to a jnp.array."""
    if isinstance(val, list):
        return jnp.array(val)
    return jnp.array([val])


def to_widths_grid(val) -> jnp.ndarray:
    """Convert width configurations to a uniform-length grid as jnp.array."""
    if isinstance(val, list):
        # Check if it's a list of lists
        if all(isinstance(v, list) for v in val):
            # It's already a nested list
            nested_list = val
        else:
            # Single list needs to be wrapped
            nested_list = [val]

        # Find max length for padding
        max_length = max(len(sublist) for sublist in nested_list)

        # Pad all sublists to the same length
        padded_lists = []
        for sublist in nested_list:
            padded = sublist + [0.0] * (max_length - len(sublist))
            padded_lists.append(padded)

        return jnp.array(padded_lists)

    raise ValueError(
        "Invalid type for widths_range conversion: expected list or nested list")


def analyzeSpectrum(params: SpectrumParams) -> Spectrum:
    widths_array = to_widths_grid(params.widths_dim)
    kappa_array = to_param_array(params.kappa_magnitude_dim)
    T_array = to_param_array(params.T_dim)
    wavelength_array = to_param_array(params.wavelength_dim)
    A0_array = to_param_array(params.A0_dim)
    B0_array = to_param_array(params.B0_dim)

    # meshgrid preparation
    kappa, T, wavelength, A0, B0 = jnp.meshgrid(
        kappa_array,
        T_array,
        wavelength_array,
        A0_array,
        B0_array,
        indexing='ij'
    )

    phase_mismatch_fn = usePPMgOSLT(wavelength, T)

    def solve_single_widths(widths):
        return solve_ncme(NCMEParams(
            fund_wave_power=A0,
            sh_wave_power=B0,
            kappa_magnitude=kappa,
            phase_mismatch_fn=phase_mismatch_fn,
            widths=widths
        ))

    mapped_solve = jax.vmap(solve_single_widths)
    results = mapped_solve(widths_array)

    return results
