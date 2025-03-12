import jax
from .device import *
from .mgoslt import *
from .solver import *
from .analyze import *


# GPUを使用するための設定
def use_gpu():
    try:
        jax.config.update('jax_platform_name', 'gpu')
        print(f"GPUを使用します: {jax.devices('gpu')}")
        return True
    except:
        print("GPUが見つかりませんでした。CPUを使用します。")
        return False
