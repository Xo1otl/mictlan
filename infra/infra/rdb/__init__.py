from .env import *

CONTAINER_NAME = 'mysql'
PORT = 3306
ADDR = f"{CONTAINER_NAME}:{PORT}"


def docker_compose():
    from .docker_compose import docker_compose
    return docker_compose
