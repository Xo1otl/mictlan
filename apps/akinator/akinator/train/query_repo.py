from typing import TypedDict, List, Dict, Protocol


class OptionsQuestions(TypedDict):
    options: List[str]
    questions: List[str]


class QueryRepo(Protocol):
    def options_questions(self) -> Dict[str, OptionsQuestions]:
        """genre_idをキーとして、そのジャンルに対する質問と選択肢のリストを返す"""
        ...

    def cases(self) -> Dict[str, List[str]]:
        """genre_idをキーとして、そのジャンルに対する場合のリストを返す"""
        ...
