from typing import Callable, Tuple, Awaitable

# --- Strategy定義 ---
# NOTE: Python 3.12以降で利用可能なGenericsの文法(PEP 695)を使用
type ProposeFn[SearchState, Query, Context] = \
    Callable[[SearchState], Awaitable[Tuple[Query, Context]]]
type ObserveFn[Query, Evidence] = Callable[[Query], Awaitable[Evidence]]
type PropagateFn[SearchState, Query, Context, Evidence] = \
    Callable[[SearchState, Query, Context, Evidence], SearchState]
type TerminationStrategy[SearchState] = Callable[[SearchState], bool]

type OrchestrateFn[SearchState] = \
    Callable[[], Awaitable[SearchState]]


def new_orchestrate[SearchState, Query, Context, Evidence](
    initial_state: SearchState,
    propose: ProposeFn[SearchState, Query, Context],
    observe: ObserveFn[Query, Evidence],
    propagate: PropagateFn[SearchState, Query, Context, Evidence],
    termination_strategy: TerminationStrategy[SearchState],
) -> OrchestrateFn[SearchState]:
    async def orchestrate() -> SearchState:
        state = initial_state
        while not termination_strategy(state):
            query, context = await propose(state)
            evidence = await observe(query)
            state = propagate(state, query, context, evidence)
        return state

    return orchestrate
