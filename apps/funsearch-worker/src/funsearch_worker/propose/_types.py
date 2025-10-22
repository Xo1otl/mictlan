from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Request[T]:
    parents: list[T]
    specification: str


@dataclass(frozen=True)
class Response:
    hypothesises: list[str]


type HandlerFunc[T] = Callable[[Request[T]], Response]
