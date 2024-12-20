from akinator.qa import Dataset
from akinator.qa.command_repo import HistoryItem
from mysql.connector import MySQLConnection
from .query_repo import QueryRepo
from .command_repo import CommandRepo
from typing import Dict, List
import uuid


class MysqlRepo(QueryRepo, CommandRepo):
    def __init__(self, conn: MySQLConnection) -> None:
        self.conn: MySQLConnection = conn

    def categories(self) -> Dict[str, str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT category_name FROM categories")
        categories = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return categories  # type: ignore

    def dataset(self, category) -> Dataset:
        cursor = self.conn.cursor()

        # choices_by_category_name テーブルからデータを取得
        cursor.execute(
            "SELECT choice_name FROM choices_with_category_name where category_name = %s", (category,))
        choices = cursor.fetchall()

        # p_case テーブルからデータを取得
        cursor.execute(
            "SELECT case_name, p_case FROM p_case where category_name = %s", (category,))
        p_case_data = cursor.fetchall()

        # p_choice_given_case_question テーブルからデータを取得
        cursor.execute(
            "SELECT case_name, question_text, choice_name, probability FROM p_choice_given_case_question where category_name = %s", (category,))
        p_choice_data = cursor.fetchall()

        cursor.close()

        dataset: Dataset = {
            "choices": [],
            "p_case": {},
            "p_choice_given_case_question": {}
        }
        # choices のデータを dataset に格納
        dataset["choices"] = [choice[0] for choice in choices]  # type: ignore

        # p_case のデータを dataset に格納
        dataset["p_case"] = {}
        for case_name, p_case in p_case_data:
            dataset["p_case"][case_name] = float(p_case)  # type: ignore

        # p_choice_given_case_question のデータを dataset に格納
        dataset["p_choice_given_case_question"] = {}
        for case_name, question_text, choice_name, probability in p_choice_data:
            if question_text not in dataset["p_choice_given_case_question"]:
                dataset[
                    "p_choice_given_case_question"][question_text] = {}  # type: ignore
            if case_name not in dataset[
                    "p_choice_given_case_question"][question_text]:  # type: ignore
                dataset[
                    "p_choice_given_case_question"][question_text][case_name] = {}  # type: ignore
            dataset[
                "p_choice_given_case_question"][question_text][case_name][choice_name] = float(probability)  # type: ignore

        return dataset

    def send_answer(self, category: str, answer: str, history: List[HistoryItem]) -> None:
        cursor = self.conn.cursor(buffered=True)

        try:
            # Get category_id
            cursor.execute(
                "SELECT category_id FROM categories WHERE category_name = %s", (category,))
            category_result = cursor.fetchone()
            if category_result is None:
                raise ValueError(f"Category '{category}' not found.")
            category_id = category_result[0]

            # Get case_id
            cursor.execute("SELECT case_id FROM cases WHERE category_id = %s AND case_name = %s",
                           (category_id, answer))  # type: ignore
            case_result = cursor.fetchone()
            if case_result is None:
                raise ValueError(
                    f"Case '{answer}' not found for category '{category}'.")
            case_id = case_result[0]

            # Prepare data for insertion
            insert_data = []
            for item in history:
                question_text = item['question']
                choice_name = item['choice']

                # Get question_id
                cursor.execute("SELECT question_id FROM questions WHERE category_id = %s AND question_text = %s", (
                    category_id, question_text))  # type: ignore
                question_result = cursor.fetchone()
                if question_result is None:
                    raise ValueError(
                        f"Question '{question_text}' not found for category '{category}'.")
                question_id = question_result[0]

                # Get choice_id
                cursor.execute("SELECT choice_id FROM choices WHERE category_id = %s AND choice_name = %s", (
                    category_id, choice_name))  # type: ignore
                choice_result = cursor.fetchone()
                if choice_result is None:
                    raise ValueError(
                        f"Choice '{choice_name}' not found for category '{category}'.")
                choice_id = choice_result[0]

                insert_data.append(
                    (str(uuid.uuid4()), case_id, question_id, choice_id))

            # Insert case_question_choices
            if insert_data:
                cursor.executemany(
                    "INSERT INTO case_question_choices (case_question_choice_id, case_id, question_id, choice_id) VALUES (%s, %s, %s, %s)", insert_data)
                self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            raise e
        finally:
            cursor.close()
