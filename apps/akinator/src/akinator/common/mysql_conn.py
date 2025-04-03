import mysql.connector
from mysql.connector import MySQLConnection
import os
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


def get_mysql_conn(host: str, user: str, password: str, database: str) -> MySQLConnection:
    conn: MySQLConnection = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )  # type: ignore
    return conn


def default_conn() -> MySQLConnection:
    print(
        f"connection opened to mysql user: {akiconf.MYSQL_USER}, db: {akiconf.MYSQL_DB}")
    return get_mysql_conn(
        host="mysql",
        user=akiconf.MYSQL_USER,
        password=akiconf.MYSQL_PASSWORD,
        database=akiconf.MYSQL_DB
    )
