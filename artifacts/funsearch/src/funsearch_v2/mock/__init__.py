import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

# ------------------------------------------------------------------------------
# 定数
# ------------------------------------------------------------------------------
GENE_LENGTH = 100
POPULATION_SIZE = 50
MAX_GENERATIONS = 100
ELITE_SIZE = 2
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.9

# ------------------------------------------------------------------------------
# 型定義
# ------------------------------------------------------------------------------
Individual = List[int]
StepSummary = Dict[str, Any]


# ------------------------------------------------------------------------------
# Context: 状態保持 (optaxの `OptState` や `params` に相当)
# ------------------------------------------------------------------------------
@dataclass
class GAContext:
    """
    GAの探索状態を保持するクラス。
    optaxにおける `params` と `OptState` を合わせたものに相当します。
    """
    generation: int
    # 現世代で評価済みのエリート個体群
    scored_population: List[Tuple[Individual, float]
                            ] = field(default_factory=list)
    # 次に評価されるべき新個体群
    population_to_evaluate: List[Individual] = field(default_factory=list)
    # 各世代のサマリー情報
    summary: StepSummary = field(default_factory=dict)


# ------------------------------------------------------------------------------
# Evaluator: 評価ロジック (optaxの `grad_fn` に相当)
# ------------------------------------------------------------------------------
class OneMaxEvaluator:
    """
    個体を評価し、スコアを算出するクラス。
    状態を変更せず、入力に対して純粋な評価値を返す責務を持ちます。
    """

    def evaluate(self, individual: Individual) -> float:
        return float(sum(individual))


# ------------------------------------------------------------------------------
# Generator: 個体生成ロジック (交叉・突然変異など)
# ------------------------------------------------------------------------------
class GAGenerator:
    """
    選択・交叉・突然変異といった遺伝的操作を通じて、次世代の個体群を生成するクラス。
    """

    def __init__(self, mutation_rate: float, crossover_rate: float, tournament_size: int, elite_size: int):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size

    def _selection(self, scored_population: List[Tuple[Individual, float]]) -> Individual:
        """トーナメント選択により親個体を1つ選出する"""
        tournament_contenders = random.sample(
            scored_population, self.tournament_size)
        return max(tournament_contenders, key=lambda item: item[1])[0]

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """二点交叉により子個体を生成する"""
        if random.random() >= self.crossover_rate:
            return parent1[:], parent2[:]
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def _mutation(self, individual: Individual) -> Individual:
        """突然変異を適用する"""
        mutated_individual = individual[:]
        for i in range(len(mutated_individual)):
            if random.random() < self.mutation_rate:
                mutated_individual[i] = 1 - mutated_individual[i]
        return mutated_individual

    def generate_next_population(self, scored_population: List[Tuple[Individual, float]]) -> Tuple[List[Tuple[Individual, float]], List[Individual]]:
        """
        評価済み個体群から、次世代のエリート個体と未評価の新個体を生成する。
        """
        sorted_population = sorted(
            scored_population, key=lambda item: item[1], reverse=True)

        # 評価済みのエリート個体
        next_elites = sorted_population[:self.elite_size]

        # 未評価の新個体
        next_new_individuals = []
        while len(next_elites) + len(next_new_individuals) < POPULATION_SIZE:
            parent1 = self._selection(scored_population)
            parent2 = self._selection(scored_population)
            child1, child2 = self._crossover(parent1, parent2)
            next_new_individuals.append(self._mutation(child1))
            if len(next_elites) + len(next_new_individuals) < POPULATION_SIZE:
                next_new_individuals.append(self._mutation(child2))

        return next_elites, next_new_individuals[:POPULATION_SIZE - len(next_elites)]


# ------------------------------------------------------------------------------
# Strategy: 次世代生成アルゴリズム (optaxの `optimizer.update` に相当)
# ------------------------------------------------------------------------------
class GAStrategy:
    """
    状態(context)と評価結果(newly_scored)を受け取り、
    次の状態を計算して返すクラス。状態の更新そのものは行わない。
    """

    def __init__(self, generator: GAGenerator):
        self.generator = generator

    def step(self, context: GAContext, newly_scored: List[Tuple[Individual, float]]) -> Dict[str, Any]:
        """
        現在の状態と、新たに評価された個体のスコア("grad")に基づき、
        次世代の状態を記述した辞書("updates")を計算して返す。
        """
        # 現世代のエリート個体と、新たに評価された個体を結合し、現世代の完全な評価済み集団を構成
        full_scored_population = context.scored_population + newly_scored

        # 現世代のサマリーを計算
        scores = [score for _, score in full_scored_population]
        best_score = max(scores) if scores else 0.0
        summary = {"generation": context.generation, "best_score": best_score}

        # 現世代の評価済み集団から、次世代のエリートと新個体を生成
        next_elites, next_new_individuals = self.generator.generate_next_population(
            full_scored_population)

        # 次世代の状態を定義する
        updates = {
            "scored_population": next_elites,
            "population_to_evaluate": next_new_individuals,
            "generation": context.generation + 1,
            "summary": summary,
        }
        return updates


# ------------------------------------------------------------------------------
# Runner: 実行エンジン
# ------------------------------------------------------------------------------
class Runner:
    """
    GAの実行フロー全体を管理するクラス。
    """

    def _apply_updates(self, context: GAContext, updates: Dict[str, Any]):
        """
        計算された更新内容(`updates`)を状態(`context`)に適用する。
        optaxの `apply_updates` に相当。
        """
        for key, value in updates.items():
            setattr(context, key, value)

    def run(self, evaluator: OneMaxEvaluator, strategy: GAStrategy, context: GAContext, max_generations: int, target_score: float):
        print(f"--- 探索開始 ---")
        start_time = time.time()

        while True:
            # --- 1. 評価 (optaxの grad計算 に相当) ---
            # 未評価の個体を評価し、スコア("grad")を計算する。
            # このステップではcontext(状態)は一切変更されない。
            newly_scored = [
                (ind, evaluator.evaluate(ind)) for ind in context.population_to_evaluate
            ]

            # --- 2. 更新内容の計算 (optaxの optimizer.update に相当) ---
            # 現状(context)と評価結果(newly_scored)から、次の状態を計算する。
            updates = strategy.step(context, newly_scored)

            # --- 3. 適用 (optaxの apply_updates に相当) ---
            # 計算された更新内容をアトミックに適用し、状態を次世代に進める。
            # 状態が変更されるのは、このメソッド内のみ。
            self._apply_updates(context, updates)

            # --- 4. ログ出力・終了判定 ---
            # 更新された最新の状態を用いてログ出力と終了判定を行う。
            best_score = context.summary.get("best_score", 0.0)
            print(
                f"世代: {context.generation - 1:03d} | "  # 表示する世代は完了した世代(更新前の世代)
                f"ベストスコア: {best_score:.0f}/{target_score:.0f}"
            )

            if best_score >= target_score or (context.generation) >= max_generations:
                if best_score >= target_score:
                    print("\n最適解に到達しました。")
                break

        end_time = time.time()
        final_best_score = context.summary.get("best_score", 0.0)
        print("\n--- 探索終了 ---")
        print(f"実行時間: {end_time - start_time:.2f} 秒")
        print(f"最終世代: {context.generation - 1}")
        print(f"最終ベストスコア: {final_best_score:.0f}")
        return context


# ------------------------------------------------------------------------------
# Controller: 全体の設定と実行
# ------------------------------------------------------------------------------
def main_controller():
    print("--- Generative Ansatz Search (GAS) PoC: optax-inspired Design ---")

    # 各コンポーネントの初期化
    evaluator = OneMaxEvaluator()
    generator = GAGenerator(mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                            tournament_size=TOURNAMENT_SIZE, elite_size=ELITE_SIZE)
    strategy = GAStrategy(generator=generator)
    runner = Runner()

    # 初期個体群の生成
    initial_population = [[random.randint(0, 1) for _ in range(
        GENE_LENGTH)] for _ in range(POPULATION_SIZE)]

    # 初期状態の定義
    initial_context = GAContext(
        generation=0,
        scored_population=[],
        population_to_evaluate=initial_population
    )

    # 実行
    runner.run(evaluator=evaluator, strategy=strategy, context=initial_context,
               max_generations=MAX_GENERATIONS, target_score=float(GENE_LENGTH))


if __name__ == '__main__':
    main_controller()
