from collections.abc import Callable
from typing import Protocol

from ._types import HandlerFunc, Request, Response


class PromptTemplate[T](Protocol):
    def build(self, specification: str, parents: list[T]) -> tuple[str, Callable[[str], list[str]]]: ...


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


def new_handler[T](prompt_template: PromptTemplate[T], llm: LLM) -> HandlerFunc[T]:
    def handle(request: Request[T]) -> Response:
        if not request.parents:
            msg = "No parents provided"
            raise ValueError(msg)
        prompt, parse = prompt_template.build(request.specification, request.parents)
        proposal = llm.generate(prompt)
        hypothesises = parse(proposal)
        return Response(hypothesises=hypothesises)

    return handle
