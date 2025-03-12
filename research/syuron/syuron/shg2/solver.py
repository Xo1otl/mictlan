from .use_device import *
from typing import NamedTuple, Callable
import jax.numpy as jnp

type FundPower = jnp.ndarray
type SHPower = jnp.ndarray
type KappaMagnitude = jnp.ndarray
type DomainWidths = jnp.ndarray
type EffTensor = jnp.ndarray


class NCMEParams(NamedTuple):
    """非線形結合モード方程式(NCME)を解くために必要なパラメータ。

    Attributes:
        fund_power: 基本波の初期パワー。
        sh_power: 第二高調波の初期パワー。
        kappa_magnitude: 非線形結合係数カッパの大きさ。
                        縦型擬似位相整合では、ドメイン間でカッパの符号のみが反転するため、大きさのみを保存。
        phase_mismatch_fn: 位置zにおける位相不整合量を計算する関数。
        domain_widths: 構造内のドメイン幅の配列。幅の合計がデバイスの全長。

    kappaをdomain毎に変えても計算量は変わらないはず。
    kappa_magnitude と domain_widthsのペアではなく、kappa と widths を持つ domains にした方が汎用性高いかもしれない。
    それか domain_indexを引数に取り kappa と widthを返す関数にする。
    """
    fund_power: FundPower
    sh_power: SHPower
    kappa_magnitude: KappaMagnitude
    phase_mismatch_fn: PhaseMismatchFn
    domain_widths: DomainWidths


type SolverFn = Callable[[NCMEParams], EffTensor]
