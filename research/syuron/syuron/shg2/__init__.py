import jax
from .use_device import *
from .analyze import *
from .use_mgoslt import *
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
