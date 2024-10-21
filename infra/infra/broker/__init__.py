CONTAINER_NAME = 'redpanda'
CONSOLE_CONTAINER_NAME = 'redpanda-console'
SCHEMA_REGISTRY_PORT = 8081
SCHEMA_REGISTRY_URL = f'http://{CONTAINER_NAME}:{SCHEMA_REGISTRY_PORT}'
KAFKA_PORT = 9092
KAFKA_ADDR = f'{CONTAINER_NAME}:{KAFKA_PORT}'


def docker_compose():
    from .docker_compose import docker_compose
    return docker_compose
