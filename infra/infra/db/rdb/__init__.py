from .env import *

CONTAINER_NAME = 'mysql'
PORT = 3306
ADDR = f"{CONTAINER_NAME}:{PORT}"


def compose():
    from .compose import compose
    return compose
