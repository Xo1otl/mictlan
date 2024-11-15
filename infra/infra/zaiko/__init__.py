from .env import *

DB_NAME = 'zaiko'
STOCK_CONNECTOR_FILE = 'stockconnector.yaml'


def compose():
    from .compose import compose
    return compose
