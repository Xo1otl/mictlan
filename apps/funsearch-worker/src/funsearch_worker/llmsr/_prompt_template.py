from collections.abc import Callable
from dataclasses import dataclass

from funsearch_worker import propose


@dataclass(frozen=True)
class Program:
    skeleton: str
    score: float


class PromptTemplate(propose.PromptTemplate[Program]):
    def build(self, specification: str, parents: list[Program]) -> tuple[str, Callable[[str], list[str]]]:
        prompt = f"Specification: {specification}\nParents: {parents}"
        # meaningless mock implementation
        best_parent = max(parents, key=lambda p: p.score)
        val = int(best_parent.score)
        new_skeletons = [
            str(val + 1),
            str(val + 1),
        ]
        return prompt, lambda _: new_skeletons
