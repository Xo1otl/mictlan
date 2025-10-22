from dataclasses import dataclass


@dataclass(frozen=True)
class ObserveRequest:
    skeleton: str


@dataclass(frozen=True)
class ObserveResponse:
    skeleton: str
    score: float


def handle_observe(request: ObserveRequest) -> ObserveResponse:
    skeleton = request.skeleton
    val = int(skeleton)
    score = float(val)
    return ObserveResponse(skeleton=skeleton, score=score)
