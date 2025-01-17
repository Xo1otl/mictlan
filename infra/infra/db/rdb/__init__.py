from .env import *

MYSQL_SERVICE_NAME = 'mysql'
MYSQL_PORT = 3306
MYSQL_ADDR = f"{MYSQL_SERVICE_NAME}:{MYSQL_PORT}"
POSTGRES_PORT = 5432
POSTGRES_SERVICE_NAME = 'postgres'
POSTGRES_ADDR = f"{POSTGRES_SERVICE_NAME}:5432"
PORTS = [MYSQL_PORT, POSTGRES_PORT]


def compose():
    from .mysql import mysql_service
    from .postgres import postgres_service
    return {
        'services': {
            MYSQL_SERVICE_NAME: mysql_service,
            POSTGRES_SERVICE_NAME: postgres_service
        },
        'volumes': {
            'mysql_data': None,
            'postgres_data': None,
        }
    }
