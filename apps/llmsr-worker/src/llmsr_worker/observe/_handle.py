from ._types import Request, Response


def handle(request: Request) -> Response:
    skeleton = request.skeleton
    val = int(skeleton)
    score = float(val)
    return Response(skeleton=skeleton, score=score)
