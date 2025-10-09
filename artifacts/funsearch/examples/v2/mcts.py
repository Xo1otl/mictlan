import math
import random
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Tuple
from enum import Enum, auto


# =========================================================================
# MCTS PoC for Markov Decision Process (MDP) - GridWorld
# =========================================================================

# --- I. 環境定義 (Environment Definition - GridWorld MDP) ---
class Action(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


@dataclass(frozen=True)
class GridState:
    """GridWorldの状態（エージェントの位置）。ハッシュ可能。"""
    x: int
    y: int


class GridWorldEnvironment:
    """決定論的なGridWorld環境。"""

    def __init__(self, width: int, height: int, start: GridState, goal: GridState, obstacles: set[GridState], move_cost: float, goal_reward: float):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.move_cost = move_cost
        self.goal_reward = goal_reward
        self.actions = list(Action)

    def is_terminal(self, state: GridState) -> bool:
        return state == self.goal

    def get_actions(self, state: GridState) -> list[Action]:
        if self.is_terminal(state):
            return []
        return self.actions

    def transition(self, state: GridState, action: Action) -> Tuple[GridState, float]:
        """状態遷移関数 T(s, a) -> s', r"""
        if self.is_terminal(state):
            return state, 0.0

        next_x, next_y = state.x, state.y
        if action == Action.UP:
            next_y -= 1
        elif action == Action.DOWN:
            next_y += 1
        elif action == Action.LEFT:
            next_x -= 1
        elif action == Action.RIGHT:
            next_x += 1

        if (0 <= next_x < self.width and 0 <= next_y < self.height and
                GridState(next_x, next_y) not in self.obstacles):
            next_state = GridState(next_x, next_y)
        else:
            # 壁や障害物にぶつかった場合は状態を変化させない
            next_state = state

        reward = self.goal_reward if next_state == self.goal else self.move_cost
        return next_state, reward


def new_gridworld_environment() -> GridWorldEnvironment:
    """標準的なGridWorld環境のファクトリー関数"""
    width, height = 6, 6
    start = GridState(0, 0)
    goal = GridState(5, 5)
    obstacles = set()
    for i in range(0, 4):
        obstacles.add(GridState(1, i))
    for i in range(2, 6):
        obstacles.add(GridState(3, i))
    obstacles.add(GridState(5, 3))

    return GridWorldEnvironment(width, height, start, goal, obstacles, move_cost=-2.0, goal_reward=10.0)


# --- II. 状態とデータ構造 (State & Data Structures) ---
@dataclass(frozen=True)
class Stats:
    """統計情報 (N: 訪問回数, Q: 累積報酬値)"""
    N: int = 0
    Q: float = 0.0

    @property
    def average_Q(self) -> float:
        return self.Q / self.N if self.N > 0 else 0.0


@dataclass(frozen=True)
class Query:
    """探索されたパス。(s0, a0, r1, s1, ...)"""
    states: List[GridState] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    def append_start(self, state: GridState):
        if self.states:
            raise ValueError("Start state already exists.")
        return replace(self, states=[state])

    def append_transition(self, action: Action, reward: float, next_state: GridState):
        return replace(self,
                       states=self.states + [next_state],
                       actions=self.actions + [action],
                       rewards=self.rewards + [reward])


# Context: ProposeFnでQueryを生成する際の文脈情報。
Context = Dict[str, Any]
# Evidence: ObserveFnが返す評価結果。（シミュレーション結果）
Evidence = float


@dataclass(frozen=True)
class SearchState:
    """探索プロセスの主要な状態。探索木の情報を不変オブジェクトとして保持する。"""
    iteration: int
    q_stats: Dict[Tuple[GridState, Action], Stats]
    n_stats: Dict[GridState, int]
    summary: Dict[str, Any] = field(default_factory=dict)

    def get_q_stats(self, state: GridState, action: Action) -> Stats:
        return self.q_stats.get((state, action), Stats())

    def get_n_stats(self, state: GridState) -> int:
        return self.n_stats.get(state, 0)


# --- III. コア関数のインターフェース定義 (Design-Aligned) ---
ProposeFn = Callable[[SearchState], Tuple[Query, Context]]
ObserveFn = Callable[[Query], Evidence]
PropagateFn = Callable[[Query, Context, Evidence, SearchState], SearchState]


# --- IV. コンポーネント実装 (MCTS Implementation) ---
class MCTSComponents:
    """MCTSのコアロジック（Propose, Observe, Propagate）を保持するクラス。"""

    def __init__(self, exploration_constant: float, discount_factor: float, max_depth: int, env: GridWorldEnvironment):
        self.C = exploration_constant
        self.gamma = discount_factor
        self.max_depth = max_depth
        self.env = env

    def _calculate_ucb1(self, stats: Stats, parent_N: int) -> float:
        if stats.N == 0:
            return float('inf')
        if parent_N == 0:
            return stats.average_Q
        exploitation = stats.average_Q
        exploration = self.C * math.sqrt(math.log(parent_N) / stats.N)
        return exploitation + exploration

    def propose_fn(self, search_state: SearchState) -> Tuple[Query, Context]:
        """Selection & Expansion: UCB1に基づき探索木をたどり、パス(Query)と空の文脈(Context)を生成する。"""
        current_state = self.env.start
        path = Query().append_start(current_state)
        visited_in_path = {current_state}
        depth = 0

        while not self.env.is_terminal(current_state) and depth < self.max_depth:
            actions = self.env.get_actions(current_state)

            legal_actions = []
            for action in actions:
                next_state, _ = self.env.transition(current_state, action)
                if next_state != current_state and next_state not in visited_in_path:
                    legal_actions.append(action)

            if not legal_actions:
                break

            parent_N = search_state.get_n_stats(current_state)
            best_action = max(legal_actions, key=lambda a: self._calculate_ucb1(
                search_state.get_q_stats(current_state, a), parent_N))

            is_expansion = search_state.get_q_stats(
                current_state, best_action).N == 0

            next_state, reward = self.env.transition(
                current_state, best_action)
            path = path.append_transition(best_action, reward, next_state)
            visited_in_path.add(next_state)
            current_state = next_state
            depth += 1

            if is_expansion:
                break

        return path, {}

    def observe_fn(self, path_candidate: Query) -> Evidence:
        """Simulation (Default Policy): パスの先端からランダムポリシーでロールアウトし、報酬を見積もる。"""
        current_state = path_candidate.states[-1]
        G_rollout = 0.0
        depth = len(path_candidate.states) - 1
        rollout_step = 0

        while not self.env.is_terminal(current_state) and depth < self.max_depth:
            actions = self.env.get_actions(current_state)
            if not actions:
                break
            action = random.choice(actions)
            next_state, reward = self.env.transition(current_state, action)
            G_rollout += (self.gamma ** rollout_step) * reward
            current_state = next_state
            depth += 1
            rollout_step += 1
        return G_rollout

    def propagate_fn(self, query: Query, context: Context, evidence: Evidence, search_state: SearchState) -> SearchState:
        """Backpropagation: シミュレーション結果を用いて新しいSearchStateを生成する。"""
        next_q_stats = search_state.q_stats.copy()
        next_n_stats = search_state.n_stats.copy()
        path = query
        G = evidence

        for i in range(len(path.actions) - 1, -1, -1):
            state, action, reward = path.states[i], path.actions[i], path.rewards[i]
            G = reward + self.gamma * G
            sa_pair = (state, action)
            current_q_stats = next_q_stats.get(sa_pair, Stats())
            new_q_stats = Stats(N=current_q_stats.N + 1,
                                Q=current_q_stats.Q + G)
            next_q_stats[sa_pair] = new_q_stats
            next_n_stats[state] = next_n_stats.get(state, 0) + 1

        summary = {
            "iteration": search_state.iteration + 1,
            "estimated_return": G,
        }
        return SearchState(
            iteration=search_state.iteration + 1,
            q_stats=next_q_stats,
            n_stats=next_n_stats,
            summary=summary,
        )


def new_mcts_components(exploration_constant: float, discount_factor: float, max_depth: int, environment: GridWorldEnvironment) -> Tuple[ProposeFn, ObserveFn, PropagateFn]:
    """MCTSコンポーネントのファクトリー関数。"""
    mcts = MCTSComponents(exploration_constant,
                          discount_factor, max_depth, environment)
    return mcts.propose_fn, mcts.observe_fn, mcts.propagate_fn


# --- V. 実行エンジン (Execution Engine) ---
class Orchestrator:
    """探索ループを駆動し、状態管理を一元的に行う。環境には依存しない。"""

    def run(self, propose_fn: ProposeFn, observe_fn: ObserveFn, propagate_fn: PropagateFn,
            initial_search_state: SearchState, max_iterations: int) -> SearchState:

        search_state = initial_search_state
        start_time = time.time()

        while search_state.iteration < max_iterations:
            query, context = propose_fn(search_state)
            evidence = observe_fn(query)
            search_state = propagate_fn(query, context, evidence, search_state)

            if (search_state.iteration % (max_iterations // 10 or 1) == 0) or search_state.iteration == max_iterations:
                print(
                    f"イテレーション: {search_state.iteration:04d} | "
                    f"今回の推定リターン(割引あり): {search_state.summary.get('estimated_return', 0.0):.4f}"
                )

        end_time = time.time()
        print(f"\n探索時間: {end_time - start_time:.4f} 秒")
        return search_state


# --- VI. 結果の解釈と可視化 (Interpretation & Visualization) ---
def get_best_plan(search_state: SearchState, environment: GridWorldEnvironment) -> Tuple[Query, float]:
    """探索結果から、現在の最良のプラン（行動系列）を抽出する。"""
    current_state = environment.start
    path = Query().append_start(current_state)
    total_reward = 0.0
    max_plan_depth = environment.width * environment.height

    while not environment.is_terminal(current_state) and len(path.actions) < max_plan_depth:
        actions = environment.get_actions(current_state)
        visitable_actions = [
            a for a in actions if search_state.get_q_stats(current_state, a).N > 0]
        if not visitable_actions:
            break

        best_action = max(
            visitable_actions, key=lambda a: search_state.get_q_stats(current_state, a).N)

        next_state, reward = environment.transition(current_state, best_action)
        path = path.append_transition(best_action, reward, next_state)
        total_reward += reward
        current_state = next_state
    return path, total_reward


def visualize_gridworld(env: GridWorldEnvironment, plan: Query):
    print("\n--- GridWorld Visualization ---")
    grid = [['.' for _ in range(env.width)] for _ in range(env.height)]
    for obs in env.obstacles:
        grid[obs.y][obs.x] = '#'

    for state in plan.states:
        if grid[state.y][state.x] == '.':
            grid[state.y][state.x] = '*'

    grid[env.start.y][env.start.x] = 'S'
    grid[env.goal.y][env.goal.x] = 'G'

    if plan.states and plan.states[0] == env.start:
        grid[env.start.y][env.start.x] = 'S*'
    if plan.states and plan.states[-1] == env.goal:
        grid[env.goal.y][env.goal.x] = 'G*'

    for row in grid:
        print(" ".join(f"{cell:<2}" for cell in row))
    print("-----------------------------\n")


# --- VII. エントリーポイント (Entry Point) ---
def main_controller():
    """依存性を注入し、Orchestratorを実行するエントリーポイント。"""
    print("--- GAS PoC: MCTS on MDP (GridWorld) [Design-Aligned] ---")

    SEED = 42
    MAX_ITERATIONS = 5000
    EXPLORATION_CONSTANT = 20.0
    DISCOUNT_FACTOR = 0.8
    MAX_DEPTH = 36
    random.seed(SEED)

    environment = new_gridworld_environment()
    propose_fn, observe_fn, propagate_fn = new_mcts_components(
        exploration_constant=EXPLORATION_CONSTANT,
        discount_factor=DISCOUNT_FACTOR,
        max_depth=MAX_DEPTH,
        environment=environment
    )
    orchestrator = Orchestrator()
    initial_search_state = SearchState(iteration=0, q_stats={}, n_stats={})

    print(f"\n--- 探索開始 (最大 {MAX_ITERATIONS} イテレーション) ---")
    visualize_gridworld(environment, Query())

    final_search_state = orchestrator.run(
        propose_fn=propose_fn,
        observe_fn=observe_fn,
        propagate_fn=propagate_fn,
        initial_search_state=initial_search_state,
        max_iterations=MAX_ITERATIONS,
    )

    print("\n--- 探索終了 ---")
    final_plan, final_reward = get_best_plan(final_search_state, environment)

    print(f"発見した最良報酬（割引なし）: {final_reward:.4f}")
    print(f"最良プラン (行動系列): {[a.name for a in final_plan.actions]}")
    visualize_gridworld(environment, final_plan)


if __name__ == '__main__':
    main_controller()
