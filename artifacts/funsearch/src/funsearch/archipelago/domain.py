from typing import Protocol, List, NamedTuple, Literal
from funsearch import function
from funsearch import profiler


class OnIslandsRemoved(NamedTuple):
    type: Literal["on_islands_removed"]
    payload: List['Island']


class OnIslandsRevived(NamedTuple):
    type: Literal["on_islands_revived"]
    payload: List['Island']


class OnBestIslandImproved(NamedTuple):
    type: Literal["on_best_island_improved"]
    payload: 'Island'


type EvolverEvent = OnIslandsRemoved | OnIslandsRevived | OnBestIslandImproved


class Evolver(profiler.Pluggable[EvolverEvent], Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


class OnBestFnImproved(NamedTuple):
    type: Literal["on_best_fn_improved"]
    payload: 'function.Function'


type IslandEvent = OnBestFnImproved


class Island(profiler.Pluggable[IslandEvent], Protocol):
    def best_fn(self) -> function.Function:
        ...

    # 島の変化は上位のコンポーネントがコントロールするため、変化は外部からの要求によって行う
    # これは、島の数だけ計算リソースが必要になることを避け、島を保持しながら余裕がある時だけ計算を呼び出すため
    def request_mutation(self) -> function.Function:
        ...
