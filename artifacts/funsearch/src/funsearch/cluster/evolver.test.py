from funsearch import archipelago
from funsearch import function
from funsearch import cluster
from funsearch import profiler


def test_evolver():
    mock_py_skeleton = function.MockPythonSkeleton()

    def evaluator(skeleton: function.Skeleton, arg: str):
        score = skeleton(1, 3) / len(arg)
        return score

    props = function.DefaultFunctionProps(
        mock_py_skeleton, ["A" * 10], evaluator)
    initial_fn = function.DefaultFunction(props)
    initial_fn.use_profiler(profiler.default_fn)

    engine = function.MockMutationEngine()
    engine.use_profiler(profiler.default_fn)

    config = cluster.IslandConfig(
        num_islands=4,
        num_selected_clusters=2,
        initial_fn=initial_fn,
        mutation_engine=engine,
    )

    islands = cluster.generate_islands(config)
    config = archipelago.MockEvolverConfig(
        islands=islands,
        num_parallel=3,
        reset_period=1
    )

    evolver = archipelago.spawn_mock_evolver(config)

    evolver.start()


if __name__ == '__main__':
    test_evolver()
