from . import usePPMgOSLT, solve_ncme, NCMEParams
import jax.numpy as jnp
from typing import List, NamedTuple, Union
import jax


class SpectrumParams(NamedTuple):
    domain_widths_dim: Union[List[List[float]], List[float]]
    kappa_magnitude_dim: Union[List[float], float]
    T_dim: Union[List[float], float]
    wavelength_dim: Union[List[float], float]
    fund_power_dim: Union[List[complex], complex]
    sh_power_dim: Union[List[complex], complex] = 0


Spectrum = jnp.ndarray


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
    domain_widths_array = to_widths_grid(params.domain_widths_dim)
    kappa_array = to_param_array(params.kappa_magnitude_dim)
    T_array = to_param_array(params.T_dim)
    wavelength_array = to_param_array(params.wavelength_dim)
    fund_power_array = to_param_array(params.fund_power_dim)
    sh_power_array = to_param_array(params.sh_power_dim)

    kappa, T, wavelength, fund_power, sh_power = jnp.meshgrid(
        kappa_array,
        T_array,
        wavelength_array,
        fund_power_array,
        sh_power_array,
        indexing='ij'
    )

    phase_mismatch_fn = usePPMgOSLT(wavelength, T)

    @jax.vmap
    def mapped_solve(domain_widths):
        return solve_ncme(NCMEParams(
            fund_power=fund_power,
            sh_power=sh_power,
            kappa_magnitude=kappa,
            phase_mismatch_fn=phase_mismatch_fn,
            domain_widths=domain_widths
        ))

    results = mapped_solve(domain_widths_array)

    return results
