import threading
import time
from typing import Dict

from funsearch import function
from funsearch import archipelago
from funsearch import cluster
from funsearch import profiler

type AllEvent = cluster.ClusterEvent | function.FunctionEvent | function.MutationEngineEvent | archipelago.EvolverEvent | archipelago.IslandEvent


class Profiler:
    def __init__(self):
        self.logger = profiler.default_logger
        self._evaluation_count = 0
        self._lock = threading.Lock()
        self._start_times_eval: Dict[int, float] = {}
        self._start_times_mutate: Dict[int, float] = {}

    def profile(self, event: AllEvent):
        message = ""
        body = ""
        current_time = time.perf_counter()
        # TODO: thread_id 使うの無理やりすぎるから時間があればイベントの型考え直した方がいいのかもしれん
        thread_id = threading.get_ident()

        if event.type == "on_evaluate":
            with self._lock:
                self._start_times_eval[thread_id] = current_time
        elif event.type == "on_evaluated":
            elapsed_time = -1.0
            with self._lock:
                start_time = self._start_times_eval.pop(thread_id, None)
                if start_time is not None:
                    elapsed_time = current_time - start_time
                self._evaluation_count += 1
            message = f"Evaluation finished in {elapsed_time:.4f}s. Score: {event.payload[1]}"
        # FIXME: 同期型島モデルの各サイクルごとの発火だからイベントが呼ばれたタイミングが正確に何回目なのか知る方法がない
        elif event.type == "on_best_island_improved":
            with self._lock:
                current_eval_count = self._evaluation_count
            message = "✨ Best island function improved!"
            best_fn = event.payload.best_fn()
            title = " Evaluated Function "
            padding = (60 - len(title)) // 2
            formatted_title = "=" * padding + title + \
                "=" * (60 - len(title) - padding)
            body = (f"""
{formatted_title}
{str(best_fn.skeleton())}
{'-' * 60}
Score      : {best_fn.score()}
Evaluations: {current_eval_count}
{'=' * 60}
""")
        elif event.type == "on_best_fn_improved":
            message = "Best function improved (within island)!"
            title = " Evaluated Function "
            padding = (60 - len(title)) // 2
            formatted_title = "=" * padding + title + \
                "=" * (60 - len(title) - padding)
            body = (f"""
{formatted_title}
{str(event.payload.skeleton())}
{'-' * 60}
Score: {str(event.payload.score())}
{'=' * 60}
""")
        elif event.type == "on_islands_removed":
            message = f"Removed islands: {[hex(id(island)) for island in event.payload]}"
        elif event.type == "on_islands_revived":
            message = f"Revived islands: {[hex(id(island)) for island in event.payload]}"
        elif event.type == "on_fn_added":
            message = f"New function added. Score: {event.payload.score()}"
        elif event.type == "on_fn_selected":  # これは cluster のイベント
            code_lengths_str = ", ".join(
                map(str, [len(str(fn.skeleton())) for fn in event.payload[0]]))
            message = f"Selected function from cluster. Code lengths: [{code_lengths_str}]. Score: {event.payload[1].score()}"
        elif event.type == "on_mutate":
            scores_str = ", ".join(
                map(str, [fn.score() for fn in event.payload]))
            message = f"Starting mutation from functions. Input scores: [{scores_str}]"
            with self._lock:
                self._start_times_mutate[thread_id] = current_time
        elif event.type == "on_mutated":
            elapsed_time = -1.0
            with self._lock:
                start_time = self._start_times_mutate.pop(thread_id, None)
                if start_time is not None:
                    elapsed_time = current_time - start_time
            scores_str = ", ".join(
                map(str, [fn.score() for fn in event.payload[0]]))
            message = f"Mutation finished in {elapsed_time:.4f}s. Input scores: [{scores_str}]"

        body_section = f"\n{body}" if body else ""
        message_part = f" | {message}"

        log_message = f"| {event.type:<20}{message_part}{body_section}"

        self.logger.info(log_message)
