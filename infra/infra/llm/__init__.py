PORT = 11434
OLLAMA_CONTAINER_NAME = 'ollama'
OLLAMA_URL = f'http://{OLLAMA_CONTAINER_NAME}:{PORT}'


def compose():
    from .compose import compose
    return compose
