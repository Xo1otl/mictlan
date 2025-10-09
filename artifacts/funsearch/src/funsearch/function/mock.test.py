from funsearch import function
from funsearch import profiler
import time
from typing import List


def test_mock():
    # function の準備
    mock_py_skeleton = function.MockPythonSkeleton()

    def evaluator(skeleton: function.Skeleton, arg: str):
        time.sleep(1)
        score = skeleton(1, 3) / len(arg)
        return score

    props = function.DefaultFunctionProps(
        mock_py_skeleton, ["A" * 10], evaluator)
    functions: List[function.Function] = [
        function.DefaultFunction(props) for _ in range(10)]
    for fn in functions:
        fn.use_profiler(profiler.default_fn)

    engine = function.MockMutationEngine()
    engine.use_profiler(profiler.default_fn)
    engine.mutate(functions)


if __name__ == "__main__":
    test_mock()
