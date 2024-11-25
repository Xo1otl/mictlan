from .env import *

SERVICE_NAME = 'mysql'
PORT = 3306
ADDR = f"{SERVICE_NAME}:{PORT}"


def compose():
    from .compose import compose
    return compose
