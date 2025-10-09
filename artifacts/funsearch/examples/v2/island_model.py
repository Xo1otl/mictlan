import random
import math
import time
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple, Dict

# =========================================================================
# GAS PoC: Island Model GA for Function Optimization (Rastrigin)
# =========================================================================

# --- I. 型定義とデータ構造 (Type Definitions & Data Structures) ---

# Ansatz（解候補）はfloatのリスト
type Ansatz = List[float]
# ScoredIndividualは (Ansatz, スコア) のタプル
type ScoredIndividual = Tuple[Ansatz, float]

# Queryは評価対象の仮説(Ansatz)のリストに。出自情報はContextへ分離。
type Query = List[Ansatz]
# Contextを追加。このPoCでは、各Ansatzがどの島に由来するかを示すindexのリスト。
type Context = List[int]


@dataclass(frozen=True)
class IslandState:
    """単一の島の状態。スコア付き母集団を保持する（イミュータブル）。"""
    scored_population: List[ScoredIndividual] = field(default_factory=list)


@dataclass(frozen=True)
class SearchState:
    """探索プロセスの全状態。複数の島を管理する (Originator)。"""
    generation: int
    islands: List[IslandState] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    """ObserveFnが返す評価結果。Queryに対応するスコアのリスト。"""
    scores: List[float]


ProposeFn = Callable[[SearchState], Tuple[Query, Context]]
ObserveFn = Callable[[Query], Evidence]
PropagateFn = Callable[[Query, Context, Evidence, SearchState], SearchState]


# --- II. ObserveFn: 仮説評価器 (Rastrigin Function) ---
def rastrigin(x: Ansatz) -> float:
    """Rastrigin関数。最小値は x=(0,...,0) で f(x)=0。"""
    n = len(x)
    A = 10.0
    return A * n + sum([(xi**2 - A * math.cos(2 * math.pi * xi)) for xi in x])


def observe_rastrigin_fn(query: Query) -> Evidence:
    """Rastrigin関数を用いてQueryを評価するObserveFnの実装。"""
    scores = [rastrigin(ansatz) for ansatz in query]
    return Evidence(scores=scores)


# --- III. ProposeFn: 仮説生成器 (Island Model GA) ---
class IslandGAProposer:
    """島モデルGA（実数値コーディング）に基づくProposeFnのロジック。"""

    def __init__(self, dimensions: int, search_range: Tuple[float, float], n_islands: int, population_per_island: int,
                 tournament_size: int, crossover_rate: float, blx_alpha: float,
                 mutation_rate: float, mutation_sigma: float, elite_size: int):
        self.dimensions = dimensions
        self.search_min, self.search_max = search_range
        self.n_islands = n_islands
        self.population_per_island = population_per_island
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.blx_alpha = blx_alpha
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.elite_size = elite_size

    def _selection(self, scored_population: List[ScoredIndividual]) -> Ansatz:
        k = min(self.tournament_size, len(scored_population))
        if k == 0:
            return []
        tournament = random.sample(scored_population, k)
        return min(tournament, key=lambda item: item[1])[0]

    def _crossover_blx_alpha(self, p1: Ansatz, p2: Ansatz) -> Tuple[Ansatz, Ansatz]:
        if random.random() > self.crossover_rate:
            return copy.deepcopy(p1), copy.deepcopy(p2)
        c1, c2 = [], []
        for x1, x2 in zip(p1, p2):
            d = abs(x1 - x2)
            min_x = min(x1, x2) - self.blx_alpha * d
            max_x = max(x1, x2) + self.blx_alpha * d
            min_x = max(self.search_min, min_x)
            max_x = min(self.search_max, max_x)
            if min_x > max_x:
                min_x, max_x = max_x, min_x
            c1.append(random.uniform(min_x, max_x))
            c2.append(random.uniform(min_x, max_x))
        return c1, c2

    def _mutation_gaussian(self, ind: Ansatz) -> Ansatz:
        mutated_ind = copy.deepcopy(ind)
        for i in range(len(mutated_ind)):
            if random.random() < self.mutation_rate:
                noise = random.gauss(0, self.mutation_sigma)
                mutated_ind[i] += noise
                mutated_ind[i] = max(self.search_min, min(
                    self.search_max, mutated_ind[i]))
        return mutated_ind

    # --- Propose Logic ---
    def propose_fn(self, state: SearchState) -> Tuple[Query, Context]:
        """SearchStateに基づき、次世代の評価対象(Query)とその文脈(Context)を生成する。"""
        if state.generation == 0:
            return self._initialize_population()

        query: Query = []
        context: Context = []
        for island_index, island in enumerate(state.islands):
            new_ansatze = self._evolve_island(island)
            query.extend(new_ansatze)
            context.extend([island_index] * len(new_ansatze))
        return query, context

    def _initialize_population(self) -> Tuple[Query, Context]:
        """初期母集団と、それに対応するContextを生成する。"""
        query: Query = []
        context: Context = []
        for island_index in range(self.n_islands):
            for _ in range(self.population_per_island):
                ansatz = [random.uniform(self.search_min, self.search_max)
                          for _ in range(self.dimensions)]
                query.append(ansatz)
                context.append(island_index)
        return query, context

    def _evolve_island(self, island: IslandState) -> List[Ansatz]:
        """単一の島を進化させ、新しいAnsatzのリストを返す。"""
        new_population: List[Ansatz] = []
        num_to_propose = self.population_per_island - self.elite_size
        while len(new_population) < num_to_propose:
            parent1 = self._selection(island.scored_population)
            parent2 = self._selection(island.scored_population)
            if not parent1 or not parent2:
                break
            child1, child2 = self._crossover_blx_alpha(parent1, parent2)
            new_population.append(self._mutation_gaussian(child1))
            if len(new_population) < num_to_propose:
                new_population.append(self._mutation_gaussian(child2))
        return new_population


def new_island_ga_propose_fn(**kwargs) -> ProposeFn:
    proposer = IslandGAProposer(**kwargs)
    return proposer.propose_fn


# --- IV. PropagateFn: 更新戦略 (Island Model GA) ---
class IslandGAPropagator:
    """島モデルGAの更新戦略（世代交代と移住）を実装するクラス。"""

    def __init__(self, elite_size: int, migration_interval: int, migration_size: int):
        self.elite_size = elite_size
        self.migration_interval = migration_interval
        self.migration_size = migration_size

    def propagate_fn(self, query: Query, context: Context, evidence: Evidence, search_state: SearchState) -> SearchState:
        """評価結果と現行状態から、次期SearchStateを計算する。"""
        n_islands = len(search_state.islands)
        if n_islands == 0 and context:
            n_islands = max(context) + 1

        # 1. Query, Context, Evidenceを島ごとに分配
        newly_scored_by_island = self._distribute_results(
            query, context, evidence, n_islands)
        # 2. 各島で世代交代
        next_islands = self._update_islands(
            search_state, newly_scored_by_island)
        # 3. 移住判定と実行
        migration_occurred = False
        if (search_state.generation + 1) > 0 and (search_state.generation + 1) % self.migration_interval == 0:
            next_islands = self._migration(next_islands)
            migration_occurred = True
        # 4. 統計情報更新と新しいSearchState生成
        return self._finalize_state(search_state, next_islands, migration_occurred)

    def _distribute_results(self, query: Query, context: Context, evidence: Evidence, n_islands: int) -> Dict[int, List[ScoredIndividual]]:
        """評価結果を島インデックスごとに整理する。"""
        results: Dict[int, List[ScoredIndividual]] = {
            i: [] for i in range(n_islands)}
        for ansatz, island_index, score in zip(query, context, evidence.scores):
            if island_index in results:
                results[island_index].append((ansatz, score))
        return results

    def _update_islands(self, search_state: SearchState, newly_scored_by_island: Dict[int, List[ScoredIndividual]]) -> List[IslandState]:
        next_islands = []
        n_islands = len(newly_scored_by_island)
        for i in range(n_islands):
            current_population = search_state.islands[i].scored_population if i < len(
                search_state.islands) else []
            new_population = []
            if self.elite_size > 0 and current_population:
                sorted_pop = sorted(current_population,
                                    key=lambda item: item[1])
                new_population.extend(sorted_pop[:self.elite_size])
            new_population.extend(newly_scored_by_island.get(i, []))
            next_islands.append(IslandState(scored_population=new_population))
        return next_islands

    def _migration(self, islands: List[IslandState]) -> List[IslandState]:
        n_islands = len(islands)
        if n_islands < 2 or self.migration_size <= 0:
            return islands
        migrants = []
        for island in islands:
            sorted_pop = sorted(island.scored_population,
                                key=lambda item: item[1])
            migrants.append(sorted_pop[:self.migration_size])
        new_islands = []
        for i in range(n_islands):
            current_island = islands[i]
            incoming_migrants = migrants[(i - 1) % n_islands]
            sorted_pop = sorted(
                current_island.scored_population, key=lambda item: item[1])
            remaining_size = max(0, len(sorted_pop) - self.migration_size)
            next_population = sorted_pop[:remaining_size] + incoming_migrants
            new_islands.append(IslandState(scored_population=next_population))
        return new_islands

    def _finalize_state(self, search_state: SearchState, next_islands: List[IslandState], migration_occurred: bool) -> SearchState:
        all_scores = [score for island in next_islands for _,
                      score in island.scored_population]
        best_score = min(all_scores) if all_scores else float('inf')
        avg_score = sum(all_scores) / \
            len(all_scores) if all_scores else float('inf')
        summary = {
            "generation": search_state.generation + 1,
            "best_score": best_score,
            "average_score": avg_score,
            "population_size": len(all_scores),
            "migration_occurred": migration_occurred
        }
        return SearchState(
            generation=search_state.generation + 1,
            islands=next_islands,
            summary=summary,
        )


def new_island_ga_propagate_fn(**kwargs) -> PropagateFn:
    propagator = IslandGAPropagator(**kwargs)
    return propagator.propagate_fn


# --- V. 実行エンジン (Execution Engine) ---
class Orchestrator:
    """探索ループを駆動し、インメモリのSearchStateを一元管理する。"""

    def run(self, propose_fn: ProposeFn, observe_fn: ObserveFn, propagate_fn: PropagateFn,
            initial_search_state: SearchState, max_generations: int, target_score: float):
        print(f"--- 探索開始 (最大 {max_generations} 世代) ---")
        start_time = time.time()
        search_state = initial_search_state

        while search_state.generation < max_generations:
            query, context = propose_fn(search_state)
            evidence = observe_fn(query)
            search_state = propagate_fn(query, context, evidence, search_state)

            self._log_progress(search_state, target_score)
            best_score = search_state.summary.get("best_score", float('inf'))
            if best_score <= target_score:
                print(f"\n目標スコア ({target_score}) に到達しました。")
                break
        else:
            print("\n最大世代数に到達しました。")
        self._finalize(search_state, start_time)

    def _log_progress(self, search_state: SearchState, target_score: float):
        best_score = search_state.summary.get("best_score", float('inf'))
        avg_score = search_state.summary.get("average_score", float('inf'))
        migration_flag = "*" if search_state.summary.get(
            "migration_occurred", False) else " "
        if search_state.generation % 10 == 0 or search_state.generation == 1 or migration_flag == "*":
            print(f"世代: {search_state.generation:03d}{migration_flag}| "
                  f"ベスト: {best_score:.6f} | "
                  f"平均: {avg_score:.6f} (目標: <{target_score:.6f})")

    def _finalize(self, search_state: SearchState, start_time: float):
        end_time = time.time()
        final_best_score = search_state.summary.get("best_score", float('inf'))
        best_ansatz = None
        if final_best_score != float('inf'):
            for island in search_state.islands:
                for ansatz, score in island.scored_population:
                    if abs(score - final_best_score) < 1e-9:
                        best_ansatz = ansatz
                        break
                if best_ansatz:
                    break
        print("\n--- 探索終了 ---")
        print(f"実行時間: {end_time - start_time:.2f} 秒")
        print(f"最終世代: {search_state.generation}")
        print(f"最終ベストスコア: {final_best_score:.8f}")
        if best_ansatz:
            display_len = min(len(best_ansatz), 5)
            print(
                f"最良解 (先頭{display_len}次元): {[f'{x:.4f}' for x in best_ansatz[:display_len]]}...")


# --- VI. エントリーポイント (Controller) ---
def main_controller():
    print("--- GAS PoC: Island Model GA for Rastrigin Optimization (Revised for Context) ---")
    SEED = 42
    random.seed(SEED)
    DIMENSIONS = 30
    SEARCH_RANGE = (-5.12, 5.12)
    TARGET_SCORE = 1e-6
    N_ISLANDS = 5
    POPULATION_PER_ISLAND = 50
    MAX_GENERATIONS = 1000
    MIGRATION_INTERVAL = 25
    MIGRATION_SIZE = 5
    ELITE_SIZE = 2
    TOURNAMENT_SIZE = 5
    CROSSOVER_RATE = 0.9
    BLX_ALPHA = 0.5
    MUTATION_RATE = 1.0 / DIMENSIONS
    MUTATION_SIGMA = (SEARCH_RANGE[1] - SEARCH_RANGE[0]) * 0.05
    print(
        f"\n設定: 次元数={DIMENSIONS}, 島数={N_ISLANDS}, 個体数/島={POPULATION_PER_ISLAND}")
    print(f"移住設定: 間隔={MIGRATION_INTERVAL}世代, サイズ={MIGRATION_SIZE}個体 (*マークで表示)")

    propose = new_island_ga_propose_fn(
        dimensions=DIMENSIONS, search_range=SEARCH_RANGE, n_islands=N_ISLANDS,
        population_per_island=POPULATION_PER_ISLAND, tournament_size=TOURNAMENT_SIZE,
        crossover_rate=CROSSOVER_RATE, blx_alpha=BLX_ALPHA, mutation_rate=MUTATION_RATE,
        mutation_sigma=MUTATION_SIGMA, elite_size=ELITE_SIZE
    )
    observe = observe_rastrigin_fn
    propagate = new_island_ga_propagate_fn(
        elite_size=ELITE_SIZE, migration_interval=MIGRATION_INTERVAL, migration_size=MIGRATION_SIZE
    )
    orchestrator = Orchestrator()
    initial_state = SearchState(generation=0, islands=[])
    orchestrator.run(
        propose_fn=propose, observe_fn=observe, propagate_fn=propagate,
        initial_search_state=initial_state, max_generations=MAX_GENERATIONS,
        target_score=TARGET_SCORE
    )


if __name__ == "__main__":
    main_controller()
