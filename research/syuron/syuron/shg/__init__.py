import jax
from .use_material import *
from .analyze_ncme import *
from .solve_ncme import *


# GPUを使用するための設定
def use_gpu():
    try:
        jax.config.update('jax_platform_name', 'gpu')
        print(f"GPUを使用します: {jax.devices('gpu')}")
        return True
    except:
        print("GPUが見つかりませんでした。CPUを使用します。")
        return False


def analyze(params: Params, use_material: UseMaterial) -> EffTensor:
    return analyze_ncme(params, use_material, solve_ncme)
