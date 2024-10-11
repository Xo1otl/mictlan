from .env import *

CONF_FILENAME = 'nginx.conf'


def docker_compose():
    from .docker_compose import docker_compose
    return docker_compose
