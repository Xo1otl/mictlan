from typing import List
import jax.numpy as jnp
import jax
from syuron import shg


def fixed(num_domains: int, width_dim: List[float], kappa_val: float) -> shg.DomainTensor:
    width_array = jnp.array(width_dim)
    widths = jnp.repeat(width_array[:, None], num_domains, axis=1)
    indices = jnp.arange(num_domains)
    kappas = jnp.where(indices % 2 == 0, kappa_val, -kappa_val)
    kappas = jnp.broadcast_to(kappas, (len(width_dim), num_domains))
    tensor = jnp.stack([widths, kappas], axis=-1)
    return tensor


def random(num_gratings: int, num_domains: int, kappa_val: float, min_width: float, max_width: float) -> shg.DomainTensor:
    key = jax.random.PRNGKey(42)
    random_widths = jax.random.uniform(key, shape=(
        num_gratings, num_domains), minval=min_width, maxval=max_width)
    random_widths = jnp.round(random_widths, 2)

    indices = jnp.arange(num_domains)
    kappa_vector = jnp.where(indices % 2 == 0, kappa_val, -kappa_val)
    kappa_gratings = jnp.broadcast_to(
        kappa_vector, (num_gratings, num_domains))

    tensor = jnp.stack([random_widths, kappa_gratings], axis=-1)
    return tensor


def chirped(num_domains: int, start_width_dim: List[float], kappa_val: float, chirp_rate_dim: List[float]) -> shg.DomainTensor:
    domain_idx_grid = jnp.arange(num_domains)
    start_width_grid = jnp.array(start_width_dim)
    chirp_rate_grid = jnp.array(chirp_rate_dim)
    domain_idx, start_width, chirp_rate = jnp.meshgrid(
        domain_idx_grid, start_width_grid, chirp_rate_grid, indexing='ij')

    widths = start_width / \
        jnp.sqrt(1 + 2 * chirp_rate * start_width * domain_idx)

    kappas = jnp.where(jnp.mod(domain_idx, 2) == 0, kappa_val, -kappa_val)
    domain_tensor = jnp.stack([widths, kappas], axis=-1)
    domain_tensor = domain_tensor.transpose((1, 2, 0, 3))
    domain_tensor = jnp.reshape(
        domain_tensor, (len(start_width_dim) * len(chirp_rate_dim), num_domains, 2))
    return domain_tensor
