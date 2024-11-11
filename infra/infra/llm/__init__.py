PORT = 11434
OLLAMA_CONTAINER_NAME = 'ollama'
OLLAMA_URL = f'http://{OLLAMA_CONTAINER_NAME}:{PORT}'


def docker_compose():
    from .docker_compose import docker_compose
    return docker_compose
