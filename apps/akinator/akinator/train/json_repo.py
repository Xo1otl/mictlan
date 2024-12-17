from typing import Dict, List
from akinator.train.query_repo import OptionsQuestions
from . import CommandRepo, QueryRepo


class JsonRepo(CommandRepo, QueryRepo):
    def add_answer(self, genre_id: str, question_id: str, case_id: str, choice: str) -> None:
        raise NotImplementedError

    def options_questions(self) -> Dict[str, OptionsQuestions]:
        raise NotImplementedError

    def cases(self) -> Dict[str, List[str]]:
        raise NotImplementedError
