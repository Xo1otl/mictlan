from .domain import *
import copy


class DefaultFunctionProps(NamedTuple):
    skeleton: 'Skeleton'
    evaluation_inputs: List
    evaluator: 'Evaluator'


# setter がなく clone によってのみ値が変更できる immutable な設計なので安心して使い回せる
class DefaultFunction(Function):
    def __init__(self, props: DefaultFunctionProps):
        self._score = None
        self._skeleton = props.skeleton
        self._evaluator = props.evaluator
        self._evaluation_inputs = props.evaluation_inputs
        self._profilers: List[Callable[[FunctionEvent], None]] = []
        self._raw_scores = None  # 新たに個別のスコアをtupleで保持する属性を追加

    def score(self):
        if self._score is None:
            raise ValueError("score is not evaluated yet")
        return self._score

    def skeleton(self):
        return self._skeleton

    def evaluate(self):
        # 基本的にimmutableとして関数の進化時などは新しいものを作るので、すでに評価済みの関数を再評価することはない
        # FIXME: 三種類のデータセットに対して evaluate をしてその平均が score で精度パターンが signature になる
        if self._score is not None:
            raise ValueError("score is already evaluated")
        for profiler_fn in self._profilers:
            profiler_fn(OnEvaluate(
                type="on_evaluate", payload=self._evaluation_inputs
            ))
        scores = [
            self._evaluator(self._skeleton, input) for input in self._evaluation_inputs
        ]
        self._raw_scores = tuple(scores)  # 個々の計算結果をtupleとして保持する
        self._score = sum(scores) / len(scores)
        for profiler_fn in self._profilers:
            profiler_fn(OnEvaluated(
                type="on_evaluated", payload=(self._evaluation_inputs, self._score)
            ))
        return self._score

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)

    def clone(self, new_skeleton=None) -> Function:
        cloned_function = copy.copy(self)
        if new_skeleton is not None:
            cloned_function._skeleton = new_skeleton
            cloned_function._score = None
        return cloned_function

    def signature(self):
        # signatureは、評価した際に得られた個々のスコアのtupleの文字列表現とする
        if self._raw_scores is None:
            raise ValueError("score is not evaluated yet")
        return str(self._raw_scores)
