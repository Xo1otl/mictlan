PORT = "3000"


def docker_compose():
    from .docker_compose import docker_compose
    return docker_compose
