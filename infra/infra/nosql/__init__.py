PORT = 27017
CONTAINER_NAME = 'mongo'


def docker_compose():
    from .docker_compose import docker_compose
    return docker_compose
