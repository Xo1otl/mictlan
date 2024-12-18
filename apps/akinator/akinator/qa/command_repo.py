from typing import Protocol, List, TypedDict


class HistoryItem(TypedDict):
    question: str
    choice: str


class CommandRepo(Protocol):
    def send_answer(self, category: str, answer: str, history: List[HistoryItem]) -> None:
        ...
