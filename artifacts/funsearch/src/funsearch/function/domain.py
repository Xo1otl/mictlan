from typing import Protocol, Callable, NamedTuple, List, Literal, Tuple, Any, List, Never
from funsearch import profiler


# Mutateの処理は時間がかかるため、処理の前後でイベントを発火する
class OnMutate(NamedTuple):
    type: Literal["on_mutate"]
    payload: List['Function']


class OnMutated(NamedTuple):
    type: Literal["on_mutated"]
    payload: Tuple[List['Function'], 'Function']


type MutationEngineEvent = OnMutate | OnMutated


class MutationEngine(profiler.Pluggable[MutationEngineEvent], Protocol):
    # 複数の関数を受け取り、それらを使って変異体を生成する
    def mutate(self, fn_list: List['Function']) -> 'Function':
        ...


# Evaluateの処理は時間がかかるため、処理の前後でイベントを発火する
class OnEvaluate(NamedTuple):
    type: Literal["on_evaluate"]
    payload: List


class OnEvaluated(NamedTuple):
    type: Literal["on_evaluated"]
    payload: Tuple[List, 'FunctionScore']


type FunctionEvent = OnEvaluate | OnEvaluated


class Skeleton[**P, R](Protocol):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        ...


class Function(profiler.Pluggable[FunctionEvent], Protocol):
    def signature(self) -> 'Signature':
        ...

    def score(self) -> 'FunctionScore':
        ...

    def skeleton(self) -> Skeleton:
        ...

    def evaluate(self) -> 'FunctionScore':
        ...

    def clone(self, new_skeleton: Skeleton | None = None) -> 'Function':
        """
        現在の Function インスタンスのクローンを返します。

        Args:
            new_skeleton: 新しい skeleton を指定した場合、クローンはこの skeleton を使用し、
                          score はリセットされます。None の場合は元の skeleton を引き継ぎ、
                          score はそのままとなります。

        Returns:
            クローンされた Function インスタンス。
        """
        ...


type Evaluator[**P, R, T] = Callable[[Skeleton[P, R], T], 'FunctionScore']
type FunctionScore = float

type Signature = str
