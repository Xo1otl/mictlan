from .env import *

DB_NAME = 'zaiko'
STOCK_CONNECTOR_FILE = 'stockconnector.yaml'


def docker_compose():
    from .docker_compose import docker_compose
    return docker_compose
