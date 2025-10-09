from typing import Callable, Protocol, Any

type Remove = Callable[[], None]


class Event(Protocol):
    @property
    def type(self) -> str: ...

    @property
    def payload(self) -> Any: ...


# 決定的でない呼び出しが多いコンポーネントで、http サーバーのように profiler を刺してイベントのプロファイリングを行う設計をするためのプロトコル
class Pluggable[T: Event](Protocol):
    def use_profiler(self, profiler_fn: 'ProfilerFn[T]') -> Remove:
        ...


type ProfilerFn[T: Event] = Callable[[T], None]
