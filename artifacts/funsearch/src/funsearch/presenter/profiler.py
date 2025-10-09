import time
import threading
import queue
from typing import Dict, Any
from funsearch import function, archipelago, cluster

AllEvent = cluster.ClusterEvent | function.FunctionEvent | function.MutationEngineEvent | archipelago.EvolverEvent | archipelago.IslandEvent


class SessionQueueProfiler:
    def __init__(self, output_queue: queue.Queue, max_mutations: int, session_data: Dict[str, Any]):
        self.q = output_queue
        self.evaluation_count = 0
        self.mutation_count = 0
        self.max_mutations = max_mutations
        self.session_data = session_data
        self.start_times_eval: Dict[int, float] = {}
        self.start_times_mutate: Dict[int, float] = {}

    def _check_stop_conditions(self) -> bool:
        # キャンセルチェックを追加
        if self.session_data.get('cancelled', False):
            self._stop_evolver()
            self.q.put(('log', "--- Process cancelled by user. ---\n"))
            self.q.put(('stop', 'Cancelled by user'))
            return True

        if self.max_mutations > 0 and self.mutation_count >= self.max_mutations:
            self._stop_evolver()
            self.q.put(
                ('log', f"--- Max mutations ({self.max_mutations}) reached. Stopping evolver. ---\n"))
            self.q.put(('stop', 'Max mutations reached'))
            return True

        return False

    def _stop_evolver(self):
        """Evolverを停止"""
        evolver = self.session_data.get('evolver')
        if evolver is not None:
            evolver.stop()
            self.session_data['evolver'] = None

    def _format_function(self, fn: Any) -> str:
        return str(fn.skeleton())

    def _get_score(self, fn_or_payload: Any) -> str:
        try:
            score = None
            if hasattr(fn_or_payload, 'score') and callable(fn_or_payload.score):
                score = fn_or_payload.score()
            elif isinstance(fn_or_payload, tuple) and len(fn_or_payload) > 1:
                score = fn_or_payload[1]
            return f"{score}" if score is not None else "N/A"
        except Exception:
            return "?.???"

    def profile(self, event: AllEvent):
        if self._check_stop_conditions():
            return

        message = ""
        body = ""
        current_time = time.perf_counter()
        thread_id = threading.get_ident()

        if event.type == "on_evaluate":
            self.start_times_eval[thread_id] = current_time
            message = "Starting evaluation..."
        elif event.type == "on_evaluated":
            elapsed_time = -1.0
            start_time = self.start_times_eval.pop(thread_id, None)
            if start_time is not None:
                elapsed_time = current_time - start_time
            self.evaluation_count += 1
            message = f"Evaluation finished in {elapsed_time:.4f}s. Score: {self._get_score(event.payload)}"
        elif event.type == "on_best_island_improved":
            count = self.evaluation_count
            message = "✨ Best island function improved!"
            best_fn = event.payload.best_fn()
            score = self._get_score(best_fn)
            code = self._format_function(best_fn)
            title = " Evaluated Function "
            padding = (60 - len(title)) // 2
            formatted_title = "=" * padding + title + \
                "=" * (60 - len(title) - padding)
            body = f"\n{formatted_title}\n{code}\n{'-' * 60}\nScore      : {score}\nEvaluations: {count}\n{'=' * 60}"
            update_message = f"**Score: {score}** (Eval: {count})\n\n```python\n{code}\n```\n\n---\n\n"
            self.q.put(('update', update_message))
        elif event.type == "on_best_fn_improved":
            count = self.evaluation_count
            message = "🏝️ Best function improved (within island)!"
            best_fn = event.payload
            score = self._get_score(best_fn)
            code = self._format_function(best_fn)
            title = " Island Best Function "
            padding = (60 - len(title)) // 2
            formatted_title = "=" * padding + title + \
                "=" * (60 - len(title) - padding)
            body = f"\n{formatted_title}\n{code}\n{'-' * 60}\nScore      : {score}\nEvaluations: {count}\n{'=' * 60}"
        elif event.type == "on_islands_removed":
            message = f"Removed islands: {[hex(id(island)) for island in event.payload]}"
        elif event.type == "on_islands_revived":
            message = f"Revived islands: {[hex(id(island)) for island in event.payload]}"
        elif event.type == "on_fn_added":
            message = f"New function added. Score: {self._get_score(event.payload)}"
        elif event.type == "on_fn_selected":
            lengths = [len(self._format_function(fn))
                       for fn in event.payload[0]]
            message = f"Selected fn. Lengths: {lengths}. Score: {self._get_score(event.payload[1])}"
        elif event.type == "on_mutate":
            self.start_times_mutate[thread_id] = current_time
            scores = [self._get_score(fn) for fn in event.payload]
            message = f"Starting mutation. Scores: {scores}"
        elif event.type == "on_mutated":
            elapsed_time = -1.0
            start_time = self.start_times_mutate.pop(thread_id, None)
            if start_time is not None:
                elapsed_time = current_time - start_time
            self.mutation_count += 1
            count = self.mutation_count
            scores = [self._get_score(fn) for fn in event.payload[0]]
            message = f"Mutation finished in {elapsed_time:.4f}s. Scores: {scores}"

        if message:
            log_message = f"| {event.type:<20} | {message}{body}\n"
            self.q.put(('log', log_message))