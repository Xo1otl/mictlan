from funsearch import archipelago
from funsearch import function
import time


def test_mock_evolver():
    mock_py_skeleton = function.MockPythonSkeleton()

    def evaluator(skeleton: function.Skeleton, arg: str):
        time.sleep(1)
        score = skeleton(1, 3) / len(arg)
        return score

    props = function.DefaultFunctionProps(
        mock_py_skeleton, ["A" * 10], evaluator)
    initial_fn = function.DefaultFunction(props)

    islands = archipelago.generate_mock_islands(
        archipelago.MockIslandConfig(num_islands=10, initial_fn=initial_fn))

    config = archipelago.MockEvolverConfig(
        islands=islands,
        num_parallel=3,
        reset_period=5
    )

    evolver = archipelago.spawn_mock_evolver(config)

    evolver.start()


if __name__ == '__main__':
    test_mock_evolver()
