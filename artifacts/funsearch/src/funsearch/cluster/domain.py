from typing import NamedTuple, List, Protocol, Literal, Tuple
from funsearch import function
from funsearch import profiler


class OnFnAdded(NamedTuple):
    type: Literal["on_fn_added"]
    payload: function.Function


class OnFnSelected(NamedTuple):
    type: Literal["on_fn_selected"]
    payload: Tuple[List[function.Function], function.Function]


type ClusterEvent = OnFnAdded | OnFnSelected


class Cluster(profiler.Pluggable[ClusterEvent], Protocol):
    def add_fn(self, fn: function.Function):
        ...

    # サンプリングの時に必要
    def select_fn(self) -> function.Function:
        ...

    # 移住の時に必要
    def best_fn(self) -> function.Function:
        ...
