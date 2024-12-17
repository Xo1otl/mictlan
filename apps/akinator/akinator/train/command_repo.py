from typing import Protocol


class CommandRepo(Protocol):
    def add_answer(self, genre_id: str, question_id: str, case_id: str, choice: str) -> None:
        """回答を追加する"""
        ...
