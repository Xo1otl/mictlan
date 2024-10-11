from .env import *


def docker_compose():
    from .docker_compose import docker_compose
    return docker_compose
