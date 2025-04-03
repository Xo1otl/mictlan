from typing import Protocol, List


class CommandRepo(Protocol):
    def edit_choices(self, category_id, choices: List[str]) -> None:
        """選択肢を追加する"""
        ...

    def add_category(self, name: str) -> str:
        """カテゴリを追加する"""
        ...

    def add_question(self, category_id: str, question_text: str) -> str:
        """質問を追加する"""
        ...

    def add_case(self, category_id: str, case_name: str) -> str:
        """場合を追加する"""
        ...

    def add_answer(self, category_id: str, question_id: str, case_id: str, choice: str) -> None:
        """回答を追加する"""
        ...
