from ._types import Request, Response


def handle(request: Request) -> Response:
    if not request.parents:
        msg = "No parents provided"
        raise ValueError(msg)

    """
    # A prompt_template is responsible for creating a prompt and a parserFunc for the LLM's output.
    # It is determined from the function signature and parent candidates.
    prompt, parse = prompt_template.build(signature, parents)

    # The single creative stochastic step where the LLM generates a proposal.
    proposal = llm.generate(prompt)

    # Create well-formed candidate solutions ready for evaluation.
    # The template's parser has all the necessary context captured.
    candidates = parse(proposal)
    """
    # **無意味なmock実装**
    best_parent = max(request.parents, key=lambda p: p.score)
    val = int(best_parent.score)

    new_skeletons = [
        str(val + 1),
        str(val + 1),
    ]
    return Response(skeletons=new_skeletons)
