from .env import *

OLLAMA_PORT = 11434
ANYTHINGLLM_PORT = 3001
PORTS = [OLLAMA_PORT, ANYTHINGLLM_PORT]
OLLAMA_SERVICE_NAME = 'ollama'
OLLAMA_URL = f'http://{OLLAMA_SERVICE_NAME}:{PORTS[0]}'


def compose():
    from .compose import compose
    return compose
