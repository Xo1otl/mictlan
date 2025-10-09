from dataclasses import dataclass


@dataclass(frozen=True)
class Program:
    skeleton: str
    score: float


@dataclass(frozen=True)
class Request:
    parents: list[Program]


@dataclass(frozen=True)
class Response:
    skeletons: list[str]
