from akinator.qa import Dataset
import mysql.connector
from mysql.connector import MySQLConnection
from mysql.connector.cursor import MySQLCursor
from infra import akinator
from .repo import Repo


class MysqlRepo(Repo):
    def __init__(self, host: str, user: str, password: str, database: str) -> None:
        self.conn: MySQLConnection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )  # type: ignore

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
            "p_case": {
            },
            "p_choice_given_case_question": {
            }
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


def default_repo() -> Repo:
    return MysqlRepo(
        host="mysql",
        user=akinator.MYSQL_USER,
        password=akinator.MYSQL_PASSWORD,
        database=akinator.MYSQL_DB
    )
