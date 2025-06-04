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


def periodical_length_limited(lengths: List[float], period: float, kappa_val: float) -> shg.DomainTensor:
    """
    指定された長さのリストに対して、周期分極反転構造を生成します。
    各長さに対して、指定の周期で分極構造を作成し、
    長さの制限内で打ち切ります。
    短いグレーティングは長さ0のドメインでパディングされます。

    Args:
        lengths: グレーティングの長さのリスト。
        period: 分極の周期 (分極の上下セットの幅)。
        kappa_val: カッパ値の絶対値。

    Returns:
        (num_gratings, max_domains, 2) の形状を持つDomainTensor。
        最後の次元は [width, kappa] です。
    """
    # 入力を JAX 配列に変換
    lengths_array = jnp.array(lengths)
    num_gratings = len(lengths)

    # 1周期内のドメイン幅（50%デューティ）
    domain_width = period / 2.0

    # 各グレーティングの最大ドメイン数を計算
    max_domains_per_grating = jnp.ceil(
        lengths_array / domain_width).astype(int)
    max_domains = jnp.max(max_domains_per_grating)

    # ドメインインデックスのグリッドを作成 (num_gratings, max_domains)
    domain_indices = jnp.arange(max_domains)[None, :]  # (1, max_domains)

    # 各ドメインの開始位置を計算 (num_gratings, max_domains)
    # (1, max_domains) -> (num_gratings, max_domains) by broadcast
    domain_starts = domain_indices * domain_width

    # 各ドメインの終了位置を計算
    domain_ends = domain_starts + domain_width

    # 各グレーティングの長さを (num_gratings, 1) に reshape してブロードキャスト
    lengths_reshaped = lengths_array[:, None]  # (num_gratings, 1)

    # ドメインが有効かどうかを判定（開始位置がグレーティング長さ未満）
    # (num_gratings, max_domains)
    valid_domains = domain_starts < lengths_reshaped

    # 各ドメインの実際の幅を計算（グレーティング長さでクリップ）
    actual_ends = jnp.minimum(domain_ends, lengths_reshaped)
    widths = jnp.where(valid_domains, actual_ends - domain_starts, 0.0)

    # カッパ値を計算（偶数インデックス: +kappa, 奇数インデックス: -kappa）
    kappas_sign = jnp.where(domain_indices %
                            2 == 0, 1.0, -1.0)  # (1, max_domains)
    kappas = jnp.where(valid_domains, kappas_sign * kappa_val,
                       0.0)  # (num_gratings, max_domains)

    # 幅とカッパ値をスタックしてテンソルを作成
    # (num_gratings, max_domains, 2)
    tensor = jnp.stack([widths, kappas], axis=-1)

    return tensor


def concatenate(domain_tensors: List[shg.DomainTensor]) -> shg.DomainTensor:
    if not domain_tensors:
        raise ValueError("domain_tensors must not be empty")

    # 最初のテンソルの形状を取得
    first_shape = domain_tensors[0].shape

    # 連結軸 (axis=1) 以外の次元の形状が一致しているか確認
    for i, tensor in enumerate(domain_tensors):
        if i == 0:  # 最初のテンソルはスキップ
            continue

        # 0番目の次元のチェック
        if tensor.shape[0] != first_shape[0]:  # type: ignore
            raise ValueError(
                f"Dimension 0 mismatch: Tensor {i} shape {tensor.shape[0]} "
                f"does not match first tensor shape"
                f"{first_shape[0]}"  # type: ignore
            )
        # 2番目以降の次元のチェック (axis=1 は連結するのでチェックしない)
        if len(tensor.shape) > 2 and len(first_shape) > 2:  # 3次元以上の場合
            if tensor.shape[2:] != first_shape[2:]:
                raise ValueError(
                    f"Dimensions from 2 onwards mismatch: Tensor {i} shape {tensor.shape[2:]} "
                    f"does not match first tensor shape {first_shape[2:]}"
                )
        elif len(tensor.shape) != len(first_shape):  # 次元数自体が異なる場合 (2次元と3次元など)
            raise ValueError(
                f"Number of dimensions mismatch: Tensor {i} has {len(tensor.shape)} dimensions, "
                f"first tensor has {len(first_shape)} dimensions."
            )

    # ドメインテンソルを連結
    concatenated_tensor = jnp.concatenate(domain_tensors, axis=1)

    return concatenated_tensor


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
