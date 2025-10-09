from .domain import *
from typing import List
import time


# 例えば llm を使った engine を作りたい時 __init__ で prompt template を渡せるようにすればよい
class MockMutationEngine(MutationEngine):
    def __init__(self):
        self._profilers: List[Callable[[MutationEngineEvent], None]] = []

    def mutate(self, fn_list: List['Function']):
        for profiler_fn in self._profilers:
            profiler_fn(OnMutate(type="on_mutate", payload=fn_list))
        # ここでは evaluate まではしない予定なので mock でも skeleton を更新して未評価にして関数を返す
        new_fn = fn_list[0].clone(fn_list[0].skeleton())
        for profiler_fn in self._profilers:
            profiler_fn(OnMutated(
                type="on_mutated",
                payload=(fn_list, new_fn)
            ))
        return new_fn

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)


class MockPythonSkeleton(Skeleton):
    def __call__(self, a: int, b: int):
        return a + b

    def __str__(self):
        return '''\
def equation_v0(x: np.ndarray, v: np.ndarray, params: np.ndarray):
    """ Mathematical function for acceleration in a damped nonlinear oscillator

    Args:
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
    """
    dv = params[0] * x  +  params[1] * v  + params[2]
    return dv
'''
