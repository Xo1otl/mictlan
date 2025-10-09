from typing import NamedTuple, List
from funsearch import profiler
from funsearch import archipelago
from funsearch import function
from funsearch import cluster
from .py_mutation_engine import PyMutationEngineUnstructured
import os
from google import genai


class EvolverConfigForMCP[**P, R, T](NamedTuple):
    equation_src: str
    docstring: str
    evaluation_inputs: List[T]
    evaluator: function.Evaluator[P, R, T]
    prompt_comment: str
    profiler_fn: profiler.ProfilerFn = profiler.default_fn
    max_nparams: int = 10
    num_islands: int = 10
    num_selected_clusters = 2
    num_parallel: int = 2
    reset_period: int = 30 * 60


def spawn_evolver_for_mcp(config: EvolverConfigForMCP) -> archipelago.Evolver:
    # function の準備
    src = config.equation_src
    py_ast_skeleton = function.PyAstSkeleton(src)
    function_props = function.DefaultFunctionProps(
        py_ast_skeleton,
        config.evaluation_inputs,
        config.evaluator
    )
    initial_fn = function.DefaultFunction(function_props)
    initial_fn.use_profiler(config.profiler_fn)
    print(f"""\
関数の初期状態を設定しました。
          
ソースコード:
```python
{src}
```
""")

    try:
        api_key = os.environ["GOOGLE_CLOUD_API_KEY"]
    except KeyError:
        from infra.ai import llm
        api_key = llm.GOOGLE_CLOUD_API_KEY
    
    gemini_client = genai.Client(api_key=api_key)
    # mutation engine の準備
    docstring = config.docstring
    mutation_engine = PyMutationEngineUnstructured(
        prompt_comment=config.prompt_comment,
        docstring=docstring or "",
        gemini_client=gemini_client,
        max_nparams=config.max_nparams
    )
    mutation_engine.use_profiler(config.profiler_fn)
    print(f'''\
変異エンジンの初期状態を設定しました。

プロンプトコメント:
"""{config.prompt_comment}"""

固定docstring:
"""
{docstring}
"""
''')

    # island の準備 (cluster への profiler の登録は evolver の init で行われる)
    islands_config = cluster.IslandConfig(
        num_islands=config.num_islands,
        num_selected_clusters=config.num_selected_clusters,
        initial_fn=initial_fn,
        mutation_engine=mutation_engine,
        island_profiler_fn=config.profiler_fn,
        cluster_profiler_fn=config.profiler_fn,
    )
    print(f"""\
島の初期状態を設定しました。
島の数: {config.num_islands}
変異時に選択されるクラスタの数: {config.num_selected_clusters}
""")

    # evolver の準備
    evolver_config = cluster.EvolverConfig(
        island_config=islands_config,
        num_parallel=config.num_parallel,
        reset_period=config.reset_period,
    )

    print(f"""\
進化者の初期状態を設定しました。
進化者の並列数: {config.num_parallel}
リセット周期: {config.reset_period} (秒)
""")

    evolver = cluster.Evolver(evolver_config)
    evolver.use_profiler(config.profiler_fn)

    return evolver
