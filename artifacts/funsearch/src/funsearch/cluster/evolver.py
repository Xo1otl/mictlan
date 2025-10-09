"""
FunSearchにおける進化アルゴリズム実行エンジン(Evolver)の実装ファイル。

このファイルでは、関数探索のための進化プロセスを管理する `Evolver` クラスと、
その構成要素である `Island` および `Cluster` クラスを定義します。

主要な特徴:
- 島モデル (Archipelago): 複数の独立した `Island`（進化集団）を並列に実行し、
  多様性を維持します。`Evolver` がこれらの `Island` を管理します。
- 関数クラスタリング: 各 `Island` 内で、関数をそのシグネチャに基づいて
  `Cluster` にグループ化します。これにより、類似した構造を持つ関数の
  集団を管理します。
- 適応的選択戦略:
    - `Island` は、クラスタのスコアと動的な温度パラメータに基づいて、
      変異の元となる関数を含むクラスタを選択します（温度は探索が進むにつれて低下）。
    - `Cluster` (`DefaultCluster` 実装) は、内部の関数から、コードの長さを
      考慮して（短いものが選ばれやすいように）関数を選択します。
- 定期的なリセット: パフォーマンスの低い `Island` を定期的にリセットし、
  最も成功している `Island` の最良関数で再初期化することで、探索の停滞を防ぎます。
- 並列処理: `Evolver` は複数の `Island` の進化ステップ（変異と評価）を
  スレッドプールを用いて並列に実行します。
"""
from typing import Callable
from funsearch import profiler
import sys
from .domain import *
from funsearch import archipelago
from funsearch import function
import time
import threading
import concurrent.futures
import traceback
from typing import List, NamedTuple
import jax
import numpy as onp
import scipy.special


class EvolverConfig(NamedTuple):
    island_config: 'IslandConfig'
    num_parallel: int
    reset_period: int


# evaluate は jax で行う予定で mutate は ollama との通信なので、両方 GIL を開放するため thread で問題ない
class Evolver(archipelago.Evolver):
    def __init__(self, config: EvolverConfig):
        self.island_config = config.island_config
        self.islands = generate_islands(self.island_config)
        self._mutation_engine = config.island_config.mutation_engine
        self._num_selected_clusters = config.island_config.num_selected_clusters
        self.num_parallel = config.num_parallel
        self.reset_period = config.reset_period
        self._profilers: List[Callable[[archipelago.EvolverEvent], None]] = []

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._active_futures: set = set()  # ★ アクティブなfutureを追跡

        if self.islands:
            self._best_score: function.FunctionScore = max(
                [island.best_fn().score() for island in self.islands])
        else:
            self._best_score: function.FunctionScore = -float('inf')

    def _reset_islands(self):
        if not self.islands or self._stop_event.is_set():
            return

        best_island = max(
            self.islands, key=lambda island: island.best_fn().score())
        sorted_islands = sorted(
            self.islands, key=lambda island: island.best_fn().score())
        num_to_reset = len(sorted_islands) // 2
        if num_to_reset == 0:
            return

        to_reset = sorted_islands[:num_to_reset]
        best_fn = best_island.best_fn()

        new_islands: List[archipelago.Island] = [Island(
            initial_fn=best_fn.clone(),
            mutation_engine=self._mutation_engine,
            num_selected_clusters=self._num_selected_clusters,
            cluster_profiler_fn=self.island_config.cluster_profiler_fn
        ) for _ in to_reset]

        for island in new_islands:
            island.use_profiler(self.island_config.island_profiler_fn)

        removed_islands = []
        new_iter = iter(new_islands)
        for idx, island in enumerate(self.islands):
            if island in to_reset:
                removed_islands.append(island)
                self.islands[idx] = next(new_iter)

        for profiler_fn in self._profilers:
            profiler_fn(archipelago.OnIslandsRemoved(
                type="on_islands_removed", payload=removed_islands))
            profiler_fn(archipelago.OnIslandsRevived(
                type="on_islands_revived", payload=new_islands))

    def _timeout_request_mutation(self, island, timeout_seconds=5.0):
        """タイムアウト付きでrequest_mutationを実行"""
        def target():
            return island.request_mutation()

        # 別スレッドで実行
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(target)

        try:
            # タイムアウト付きで結果を待つ
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            print(f"Mutation timeout for island {hex(id(island))}")
            # タイムアウトした場合はNoneを返す（futureはバックグラウンドで実行継続）
            return None
        except Exception as e:
            print(f"Mutation error for island {hex(id(island))}: {e}")
            return None
        finally:
            executor.shutdown(wait=False)

    def _evolve_islands(self):
        if self._stop_event.is_set():
            return

        print(">>> evolving islands...")

        # ★ 短いタイムアウトで強制的に停止できるようにする
        mutation_timeout = 3.0  # 3秒でタイムアウト

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_parallel)

        try:
            # ★ タイムアウト付きのmutation実行
            future_to_island = {
                self._executor.submit(self._timeout_request_mutation, island, mutation_timeout): island
                for island in self.islands
            }

            self._active_futures = set(future_to_island.keys())

            # ★ 非常に短いタイムアウトで頻繁にチェック
            while future_to_island and not self._stop_event.is_set():
                try:
                    done, not_done = concurrent.futures.wait(
                        future_to_island.keys(),
                        timeout=0.1,  # ★ 100msごとにチェック
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for future in done:
                        island = future_to_island.pop(future)
                        self._active_futures.discard(future)
                        try:
                            result = future.result()  # 既に完了しているので即座に取得
                            # 結果の処理（スコア更新など）は停止中でなければ実行
                            if result and not self._stop_event.is_set():
                                pass  # 必要に応じて結果を処理
                        except Exception as e:
                            if not self._stop_event.is_set():
                                print(
                                    f"Error processing result for island {hex(id(island))}: {e}", file=sys.stderr)

                    # ★ 停止チェック - 少しでも停止が要求されたらすぐに抜ける
                    if self._stop_event.is_set():
                        break

                except Exception as e:
                    if not self._stop_event.is_set():
                        print(
                            f"Error in thread pool wait: {e}", file=sys.stderr)
                    break

            # ★ 停止時の強制終了処理
            if self._stop_event.is_set() and future_to_island:
                print(
                    f"Force stopping {len(future_to_island)} ongoing tasks...")

        finally:
            # ★ executorの強制シャットダウン
            if self._executor:
                try:
                    # Python 3.9+
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    # Python 3.8以下
                    self._executor.shutdown(wait=False)
                self._executor = None
            self._active_futures.clear()

        # スコア更新（停止中でなければ）
        if not self._stop_event.is_set() and self.islands:
            try:
                best_island = max(
                    self.islands, key=lambda i: i.best_fn().score())
                if best_island.best_fn().score() > self._best_score:
                    self._best_score = best_island.best_fn().score()
                    for profiler_fn in self._profilers:
                        profiler_fn(archipelago.OnBestIslandImproved(
                            type="on_best_island_improved", payload=best_island))
            except Exception as e:
                print(f"Error updating best score: {e}", file=sys.stderr)

    def _run(self):
        last_reset_time = time.time()
        print("Evolution started...")

        try:
            while not self._stop_event.is_set():
                # ★ 進化ステップ（短時間で終わるか、タイムアウトで強制終了）
                self._evolve_islands()

                # ★ 停止チェック
                if self._stop_event.is_set():
                    break

                # JAXキャッシュクリア（停止チェック後）
                if not self._stop_event.is_set():
                    jax.clear_caches()

                # ★ 停止チェック
                if self._stop_event.is_set():
                    break

                # リセット処理（停止チェック付き）
                if time.time() - last_reset_time >= self.reset_period:
                    if not self._stop_event.is_set():
                        self._reset_islands()
                        last_reset_time = time.time()

                # ★ 非常に短いスリープで高い応答性を維持
                for _ in range(10):  # 0.01秒 × 10 = 0.1秒
                    if self._stop_event.is_set():
                        break
                    time.sleep(0.01)

        except Exception as e:
            if not self._stop_event.is_set():
                print(f"Evolution error: {e}", file=sys.stderr)
                traceback.print_exc()
        finally:
            print("<<< evolution stopped")

    def start(self):
        if self._thread and self._thread.is_alive():
            print("Evolver is already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        try:
            # ★ 短い間隔でチェック
            while self._thread.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Stopping evolver...")
            self.stop()
            # 短いタイムアウトで待機
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=3)
                if self._thread.is_alive():
                    print("Warning: Evolution thread did not stop gracefully")

    def stop(self) -> None:
        print("Stopping evolver...")
        self._stop_event.set()

        # ★ アクティブなfutureを強制終了（無理やり）
        if self._executor and not self._executor._shutdown:
            print("Force shutting down thread pool executor...")
            self._executor.shutdown(wait=False, cancel_futures=True)

        # ★ 短いタイムアウトで待機
        if self._thread is not None and threading.current_thread() != self._thread:
            print("Waiting for evolution thread to stop...")
            self._thread.join(timeout=2)  # ★ 2秒でタイムアウト
            if self._thread.is_alive():
                print(
                    "Warning: Evolution thread did not stop in time, but stop signal sent.")
            else:
                print("Evolution thread stopped successfully.")

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)


class IslandConfig(NamedTuple):
    num_islands: int
    num_selected_clusters: int
    initial_fn: function.Function
    mutation_engine: function.MutationEngine
    island_profiler_fn: profiler.ProfilerFn = profiler.default_fn
    cluster_profiler_fn: profiler.ProfilerFn = profiler.default_fn


def generate_islands(config: IslandConfig) -> List[archipelago.Island]:
    config.initial_fn.evaluate()
    islands: List[archipelago.Island] = []
    for _ in range(config.num_islands):
        island = Island(
            config.initial_fn, config.mutation_engine, config.num_selected_clusters, config.cluster_profiler_fn
        )
        island.use_profiler(config.island_profiler_fn)
        islands.append(island)
    return islands


class Island(archipelago.Island):
    def __init__(self, initial_fn: function.Function, mutation_engine: function.MutationEngine, num_selected_clusters: int, cluster_profiler_fn: profiler.ProfilerFn):
        self._best_fn = initial_fn
        self._mutation_engine = mutation_engine
        self._profilers: List[Callable[[archipelago.IslandEvent], None]] = []
        self._num_selected_clusters = num_selected_clusters
        self._cluster_profiler_fn = cluster_profiler_fn
        self.clusters: dict[str, Cluster] = {
            initial_fn.signature(): DefaultCluster(initial_fn)}
        for cluster in self.clusters.values():
            cluster.use_profiler(self._cluster_profiler_fn)
        self._num_fns = 0
        self._cluster_sampling_temperature_init = 0.1
        self._cluster_sampling_temperature_period = 30_000

    def _select_clusters(self) -> List[Cluster]:
        """
        スコアと温度に基づいてクラスタを選択する。
        非有限スコアはエラーとし、scipy.special.softmax を使用。
        """
        available_clusters = list(self.clusters.values())
        num_clusters = len(available_clusters)
        scores = onp.array([cluster.best_fn().score()
                           for cluster in available_clusters], dtype=float)
        if not onp.all(onp.isfinite(scores)):
            problematic_indices = onp.where(~onp.isfinite(scores))[0]
            problematic_skeletons = [str(available_clusters[idx].best_fn().skeleton())
                                     for idx in problematic_indices]
            problematic_info = ", ".join(f"index {idx}: '{skel}'"
                                         for idx, skel in zip(problematic_indices, problematic_skeletons))
            raise ValueError(
                f"Non-finite scores detected. Problematic clusters -> [{problematic_info}]")

        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * \
            (1 - (self._num_fns % period) / period)
        safe_temperature = max(temperature, float(onp.finfo(float).tiny))

        logits = scores / safe_temperature
        probabilities = scipy.special.softmax(logits, axis=-1)

        num_available_clusters = len(onp.where(probabilities > 0)[0])
        num_to_select = min(self._num_selected_clusters,
                            num_available_clusters)

        if num_to_select <= 0:
            raise ValueError("No clusters available for selection.")

        try:
            selected_indices = onp.random.choice(
                num_clusters, size=num_to_select, replace=False, p=probabilities
            )
            return [available_clusters[i] for i in selected_indices]
        except ValueError as e:
            prob_sum = onp.sum(probabilities)
            raise ValueError(
                f"Cluster selection failed in np.random.choice. Check probabilities (sum={prob_sum}, has_nan={onp.isnan(probabilities).any()}). Original error: {e}"
            ) from e

    def _move_to_cluster(self, fn: function.Function):
        signature = fn.signature()
        if signature not in self.clusters:
            new_cluster = DefaultCluster(initial_fn=fn)
            new_cluster.use_profiler(self._cluster_profiler_fn)
            self.clusters[signature] = new_cluster
        else:
            self.clusters[signature].add_fn(fn)
        self._num_fns += 1

    def request_mutation(self):
        sample_clusters = self._select_clusters()
        sample_fns = [cluster.select_fn() for cluster in sample_clusters]
        # まずここに時間がかかる
        new_fn = self._mutation_engine.mutate(sample_fns)
        # これも時間がかかる
        new_score = new_fn.evaluate()
        self._move_to_cluster(new_fn)
        if new_score > self._best_fn.score():
            self._best_fn = new_fn
            for profiler_fn in self._profilers:
                profiler_fn(archipelago.OnBestFnImproved(
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


class DefaultCluster(Cluster):
    def __init__(self, initial_fn: function.Function) -> None:
        self._functions = [initial_fn]
        self._profilers: List[Callable[[ClusterEvent], None]] = []

    def select_fn(self) -> function.Function:
        # 各関数の skeleton() からソースコードの長さを取得
        lengths = onp.array([len(str(fn.skeleton()))
                            for fn in self._functions])
        # 最小の長さを引いて正規化する（各値を (length - min) / (max + 1e-6) に変換）
        normalized_lengths = (lengths - lengths.min()) / (lengths.max() + 1e-6)
        # 短い関数が選ばれやすくなるよう、正規化した値の負数を logits とする
        logits = -normalized_lengths
        # ソフトマックス計算： exp(logits) / sum(exp(logits))
        exp_logits = onp.exp(logits)
        probabilities = exp_logits / exp_logits.sum()

        # 上記確率に従って関数を選択
        selected_fn = onp.random.choice(
            self._functions, p=probabilities)  # type: ignore

        for profiler_fn in self._profilers:
            profiler_fn(OnFnSelected(
                type="on_fn_selected", payload=(self._functions, selected_fn)
            ))
        return selected_fn

    def add_fn(self, fn: function.Function):
        # 追加する関数の signature が一致するかどうかは、呼び出し側で確認
        self._functions.append(fn)
        for profiler_fn in self._profilers:
            profiler_fn(OnFnAdded(type="on_fn_added", payload=fn))

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)

    def best_fn(self):
        return max(self._functions, key=lambda fn: fn.score())
