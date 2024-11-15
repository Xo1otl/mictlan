SERVICE_NAME = 'redpanda'
CONSOLE_SERVICE_NAME = 'redpanda-console'
SCHEMA_REGISTRY_PORT = 8081
SCHEMA_REGISTRY_URL = f'http://{SERVICE_NAME}:{SCHEMA_REGISTRY_PORT}'
KAFKA_PORT = 9092
KAFKA_ADDR = f'{SERVICE_NAME}:{KAFKA_PORT}'


def compose():
    from .compose import compose
    return compose
