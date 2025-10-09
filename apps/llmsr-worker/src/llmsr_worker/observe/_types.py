from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    skeleton: str


@dataclass(frozen=True)
class Response:
    skeleton: str
    score: float
