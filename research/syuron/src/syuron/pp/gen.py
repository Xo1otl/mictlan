from typing import List, Callable
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


def periodical(num_domains: int, period_dim: List[float], kappa_val: float, duty: float) -> shg.DomainTensor:
    """
    周期的な分極反転構造のテンソルを生成します。
    周期 (period) とデューティ比 (duty) から各ドメインの幅を計算します。

    Args:
        num_domains: ドメインの総数。
        period_dim: 周期 (分極の上下セットの幅) のリスト。
        kappa_val: カッパ値の絶対値。
        duty: デューティ比 (周期に対する最初のドメインの幅の割合)。

    Returns:
        生成されたドメインテンソル。
    """
    # period_dim を JAX 配列に変換し、ブロードキャストのために新しい軸を追加します。
    period_array = jnp.array(period_dim)[:, None]

    # 1周期内の2つのドメインの幅を計算します。
    width1 = period_array * duty
    width2 = period_array * (1.0 - duty)

    # ドメインのインデックスを作成します。
    indices = jnp.arange(num_domains)

    # ドメインのインデックスが偶数か奇数かに基づいて幅を割り当てます。
    # width1 (a, 1) と width2 (a, 1) が indices (b) とブロードキャストされ、(a, b) の形状になります。
    widths = jnp.where(indices % 2 == 0, width1, width2)

    # カッパ値を割り当て、符号を交互に変更します。
    kappas_base = jnp.where(indices % 2 == 0, kappa_val, -kappa_val)
    # カッパ値を (a, b) の形状にブロードキャストします。
    kappas = jnp.broadcast_to(kappas_base, (len(period_dim), num_domains))

    # 幅とカッパ値を最後の軸に沿ってスタックします。
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


def from_width_function(num_domains: int, kappa_val: float, width_func: Callable[[jnp.ndarray], jnp.ndarray]) -> shg.DomainTensor:
    """
    ユーザー定義の幅関数から単一のドメインテンソルを生成します。
    幅関数はドメインインデックスのみを受け取ります。
    width_func はベクトル化されるので、jaxのwhere等を使って実装する必要がある

    Args:
        num_domains: ドメインの総数。
        kappa_val: カッパ値の絶対値。
        width_func: ドメインインデックス (0 から num_domains-1) を受け取り、
                    そのドメインの幅を返す JAX 互換の関数。

    Returns:
        生成されたドメインテンソル (形状: (1, num_domains, 2))。
    """
    domain_indices = jnp.arange(num_domains)

    # width_func をドメインインデックスに対してベクトル化して適用します。
    widths_vector = jax.vmap(width_func)(domain_indices)

    # 形状を (1, b) に変更します。
    widths = widths_vector[None, :]

    # カッパ値を作成します (形状: (1, b))。
    kappa_vector = jnp.where(domain_indices % 2 == 0, kappa_val, -kappa_val)
    kappas = kappa_vector[None, :]

    # 幅とカッパ値を最後の軸に沿ってスタックします。
    tensor = jnp.stack([widths, kappas], axis=-1)

    return tensor
