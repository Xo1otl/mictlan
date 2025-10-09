from typing import Callable
from funsearch import function
from funsearch import profiler
import sys
from .domain import *
import time
import threading
import concurrent.futures
import traceback


class MockEvolverConfig(NamedTuple):
    islands: List[Island]
    num_parallel: int
    reset_period: int


def spawn_mock_evolver(config: MockEvolverConfig) -> Evolver:
    evolver = MockEvolver(config)
    evolver.use_profiler(profiler.default_fn)
    return evolver


# evaluate は jax で行う予定で mutate は ollama との通信なので、両方 GIL を開放するため thread で問題ない
class MockEvolver(Evolver):
    def __init__(self, config: MockEvolverConfig):
        self.islands = config.islands
        self.num_parallel = config.num_parallel
        self.reset_period = config.reset_period
        self._profilers: List[Callable[[EvolverEvent], None]] = []
        self.running: bool = False
        self._thread: threading.Thread | None = None
        self.best_island: Island | None = None

    def _reset_islands(self):
        # 一番良い島を取得
        best_island = max(
            self.islands, key=lambda island: island.best_fn().score())
        # 島をスコアの低い順に並べ替える
        sorted_islands = sorted(
            self.islands, key=lambda island: island.best_fn().score())
        num_to_reset = len(sorted_islands) // 2
        if num_to_reset == 0:
            return

        # 下位半分をリセット対象とする (LLM-SRと同じ基準)
        to_reset = sorted_islands[:num_to_reset]

        # 一番良い島の best_fn とスコアを使って、新しい島を生成する
        best_fn = best_island.best_fn()
        new_islands: List[Island] = [
            # skeleton も変えていないので clone された function ではスコアも保持されている
            MockIsland(initial_fn=best_fn.clone()) for _ in to_reset
        ]

        removed_islands = []
        new_iter = iter(new_islands)
        # 対象の島を新しい島に置き換える
        for idx, island in enumerate(self.islands):
            if island in to_reset:
                removed_islands.append(island)
                self.islands[idx] = next(new_iter)

        # プロファイラへのイベント通知
        for profiler_fn in self._profilers:
            profiler_fn(OnIslandsRemoved(
                type="on_islands_removed", payload=removed_islands))
            profiler_fn(OnIslandsRevived(
                type="on_islands_revived", payload=new_islands))

    def _evolve_islands(self):
        print(">>> evolving islands...")
        # Evolve islands by calling request_mutation concurrently,
        # but do not exceed the allowed parallelism.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            future_to_island = {
                executor.submit(island.request_mutation): island for island in self.islands
            }
            for future in concurrent.futures.as_completed(future_to_island):
                island = future_to_island[future]
                try:
                    _ = future.result()
                except Exception as e:
                    # In a mock, simply ignore mutation errors.
                    print(f"Error during mutation: {e}", file=sys.stderr)
                    traceback.print_exc()
                    # エラーを見たいので止めるようにする
                    # self.stop()
                    continue
                # If this island now has a higher score than any previous best,
                # update best_island and trigger the on_best_island_improved event.
                if self.best_island is None or island.best_fn().score() > self.best_island.best_fn().score():
                    self.best_island = island
                    for profiler_fn in self._profilers:
                        profiler_fn(OnBestIslandImproved(
                            type="on_best_island_improved", payload=island))

    def _run(self):
        last_reset_time = time.time()
        while self.running:
            # TODO: 並列で処理するけど、全部一斉に終わらないと次に行かない設計になってる、ちょっと効率悪いから、時間があれば直そう
            # Go の context みたいなのがあれば安全に実装できそうだが、完璧な実装はかなり大変そう
            self._evolve_islands()
            # Check if it's time to reset low-scoring islands.
            if time.time() - last_reset_time >= self.reset_period:
                self._reset_islands()
                last_reset_time = time.time()
        print("<<< evolution stopped")

    def start(self):
        # Begin evolution in a background thread.
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping evolver... Waiting for threads to finish.")
            self.stop()

    def stop(self) -> None:
        # Signal the thread to stop and wait for it to finish.
        self.running = False
        if self._thread is not None and threading.current_thread() != self._thread:
            self._thread.join()

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)


class MockIslandConfig(NamedTuple):
    num_islands: int
    initial_fn: function.Function


def generate_mock_islands(config: MockIslandConfig) -> List[Island]:
    config.initial_fn.evaluate()
    islands: List[Island] = [
        MockIsland(config.initial_fn) for _ in range(config.num_islands)
    ]
    # TODO: 必要なlistenerの登録など
    return islands


class MockIsland(Island):
    def __init__(self, initial_fn: function.Function):
        self._best_fn = initial_fn
        self._profilers: List[Callable[[IslandEvent], None]] = []

    def request_mutation(self):
        time.sleep(3)
        # evaluate するには skeleton を置き換えて score をリセットする必要がある
        new_fn = self._best_fn.clone(self._best_fn.skeleton())
        new_score = new_fn.evaluate()
        # score を見て new_fn を適切な cluster に移動する処理が必要だけど、それは cluster モジュールの cluster 依存の島で実装すればよい
        if new_score > self._best_fn.score():
            # 最良の関数のスコアをそのまま島のスコアとする (LLM-SRと同じ基準)
            self._best_fn = new_fn
            for profiler_fn in self._profilers:
                profiler_fn(OnBestFnImproved(
                    type="on_best_fn_improved",
                    payload=new_fn
                ))
        return new_fn

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)

    def best_fn(self) -> function.Function:
        if self._best_fn is None:
            raise ValueError("best_fn not set")
        return self._best_fn
