from typing import Dict, List
from . import CommandRepo, QueryRepo, Category
import mysql.connector
from mysql.connector import MySQLConnection
from mysql.connector.cursor import MySQLCursor
import uuid
import os


class MysqlRepo(CommandRepo, QueryRepo):
    def __init__(self, host: str, user: str, password: str, database: str) -> None:
        self.conn: MySQLConnection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )  # type: ignore

    def edit_choices(self, category_id, choices: List[str]) -> None:
        cursor: MySQLCursor = self.conn.cursor(dictionary=True)

        try:
            # Delete existing choices for the category
            cursor.execute(
                "DELETE FROM choices WHERE category_id = %s", (category_id,)
            )

            # Insert new choices
            for choice in choices:
                cursor.execute(
                    "INSERT INTO choices (category_id, choice_name) VALUES (%s, %s)",
                    (category_id, choice)
                )

            self.conn.commit()
        except mysql.connector.Error as err:
            self.conn.rollback()
            raise err
        finally:
            cursor.close()

    def add_category(self, name: str) -> str:
        cursor: MySQLCursor = self.conn.cursor(dictionary=True)
        id = str(uuid.uuid4())
        try:
            cursor.execute(
                "INSERT INTO categories (category_id, category_name) VALUES (%s, %s)", (
                    id, name)
            )
            self.conn.commit()
        except mysql.connector.Error as err:
            self.conn.rollback()
            raise err
        finally:
            cursor.close()
        return id

    def add_question(self, category_id: str, question_text: str) -> str:
        cursor: MySQLCursor = self.conn.cursor(dictionary=True)
        id = str(uuid.uuid4())

        # Insert the question
        try:
            cursor.execute(
                "INSERT INTO questions (question_id, category_id, question_text) VALUES (%s, %s, %s)",
                (id, category_id, question_text)
            )
            self.conn.commit()
        except mysql.connector.Error as err:
            self.conn.rollback()
            raise err
        finally:
            cursor.close()
        return id

    def add_case(self, category_id: str, case_name: str) -> str:
        cursor: MySQLCursor = self.conn.cursor(dictionary=True)
        id = str(uuid.uuid4())

        # Insert the case
        try:
            cursor.execute(
                "INSERT INTO cases (case_id, category_id, case_name) VALUES (%s, %s, %s)",
                (id, category_id, case_name)
            )
            self.conn.commit()
        except mysql.connector.Error as err:
            self.conn.rollback()
            raise err
        finally:
            cursor.close()
        return id

    def add_answer(self, category_id: str, question_id: str, case_id: str, choice: str) -> None:
        cursor: MySQLCursor = self.conn.cursor(dictionary=True)

        # Get choice_id based on genre_id and choice name
        cursor.execute(
            "SELECT choice_id FROM choices WHERE category_id = %s AND choice_name = %s",
            (category_id, choice)
        )
        choice_data = cursor.fetchone()
        if choice_data is None:
            cursor.close()
            raise ValueError(
                f"Choice '{choice}' not found for genre '{category_id}'")

        choice_id = choice_data["choice_id"]  # type: ignore

        # Insert the answer
        try:
            cursor.execute(
                "INSERT INTO case_question_choices (case_question_choice_id, case_id, question_id, choice_id) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), case_id, question_id, choice_id)
            )
            self.conn.commit()
        except mysql.connector.Error as err:
            self.conn.rollback()
            raise err
        finally:
            cursor.close()

    def categories(self) -> Dict[str, Category]:
        cursor: MySQLCursor = self.conn.cursor(dictionary=True)

        # Fetch categories
        cursor.execute("SELECT category_id, category_name FROM categories")
        categories_data = cursor.fetchall()

        categories: Dict[str, Category] = {}
        for cat_data in categories_data:
            category_id = cat_data["category_id"]  # type: ignore
            categories[category_id] = {
                "text": cat_data["category_name"],  # type: ignore
                "choices": [],
                "cases": {},
                "questions": {}
            }

        # Fetch choices for each category
        cursor.execute("SELECT category_id, choice_name FROM choices")
        choices_data = cursor.fetchall()
        for choice_data in choices_data:
            categories[choice_data["category_id"]]["choices"].append(  # type: ignore
                choice_data["choice_name"])  # type: ignore

        # Fetch cases for each category
        cursor.execute("SELECT case_id, category_id, case_name FROM cases")
        cases_data = cursor.fetchall()
        for case_data in cases_data:
            categories[case_data["category_id"]]["cases"][case_data["case_id"]] = {  # type: ignore
                "text": case_data["case_name"]  # type: ignore
            }

        # Fetch questions for each category
        cursor.execute(
            "SELECT question_id, category_id, question_text FROM questions")
        questions_data = cursor.fetchall()
        for question_data in questions_data:
            categories[question_data["category_id"]]["questions"][question_data["question_id"]] = {  # type: ignore
                "text": question_data["question_text"]  # type: ignore
            }

        cursor.close()
        return categories


try:
    import infra.akinator as akiconf
except ImportError:
    # infra モジュールが存在しない場合、環境変数から設定を読み込む
    class AkinatorConfig:
        def __init__(self):
            self.MYSQL_USER = os.environ.get("MYSQL_USER", "user")
            self.MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "password")
            self.MYSQL_DB = os.environ.get("MYSQL_DB", "akinator_db")

    akiconf = AkinatorConfig()


def default_repo() -> MysqlRepo:
    return MysqlRepo(
        host="mysql",
        user=akiconf.MYSQL_USER,
        password=akiconf.MYSQL_PASSWORD,
        database=akiconf.MYSQL_DB
    )
