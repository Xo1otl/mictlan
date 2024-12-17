from typing import TypedDict, List, Dict, Protocol


class Case(TypedDict):
    text: str


class Question(TypedDict):
    text: str


class Category(TypedDict):
    text: str
    choices: List[str]
    cases: Dict[str, Case]
    questions: Dict[str, Question]


class QueryRepo(Protocol):
    def categories(self) -> Dict[str, Category]:
        ...
