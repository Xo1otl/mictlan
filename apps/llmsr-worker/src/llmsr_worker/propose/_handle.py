from ._types import Request, Response


def handle(request: Request) -> Response:
    if not request.parents:
        msg = "No parents provided"
        raise ValueError(msg)

    """
    1. llmsrの場合はskeletonのシグネチャがベースとなるかな
    2. alpha evolveの場合はbaseが元のプログラム

    original, query = prompt_sampler.build(request.parents)
    diffs = llm.generate(query)
    for diff in diffs:
        skeleton = original.apply(diff)
        new_skeletons.append(skeleton)
    """
    best_parent = max(request.parents, key=lambda p: p.score)
    val = int(best_parent.score)

    new_skeletons = [
        str(val + 1),
        str(val + 1),
    ]
    return Response(skeletons=new_skeletons)
