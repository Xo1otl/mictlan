from . import *

# Redpandaサービスの設定
redpanda_service = {
    'command': [
        SERVICE_NAME,
        'start',
        f'--kafka-addr internal://0.0.0.0:{KAFKA_PORT},external://0.0.0.0:19092',
        f'--advertise-kafka-addr internal://{SERVICE_NAME}:9092,external://localhost:19092',
        '--pandaproxy-addr internal://0.0.0.0:8082,external://0.0.0.0:18082',
        f'--advertise-pandaproxy-addr internal://{SERVICE_NAME}:8082,external://localhost:18082',
        f'--schema-registry-addr internal://0.0.0.0:{SCHEMA_REGISTRY_PORT},external://0.0.0.0:18081',
        f'--rpc-addr {SERVICE_NAME}:33145',
        f'--advertise-rpc-addr {SERVICE_NAME}:33145',
        '--mode dev-container',
        '--smp 1',
        '--default-log-level=info'
    ],
    'image': 'docker.redpanda.com/redpandadata/redpanda:latest',
    'container_name': SERVICE_NAME,
    'volumes': [
        'redpanda:/var/lib/redpanda/data'
    ],
    'ports': [
        '18081:18081',
        '18082:18082',
        '19092:19092',
        '19644:9644'
    ]
}

console_config_content = f'''kafka:
  brokers: ["{KAFKA_ADDR}"]
  schemaRegistry:
    enabled: true
    urls: ["{SCHEMA_REGISTRY_URL}"]
redpanda:
  adminApi:
    enabled: true
    urls: ["http://redpanda:9644"]'''

# Redpanda Consoleサービスの設定
redpanda_console_service = {
    'container_name': CONSOLE_SERVICE_NAME,
    'image': 'docker.redpanda.com/redpandadata/console:latest',
    'entrypoint': '/bin/sh',
    'command': '-c \'echo "$$CONSOLE_CONFIG_FILE" > /tmp/config.yml; /app/console\'',
    'environment': {
        'CONFIG_FILEPATH': '/tmp/config.yml',
        'CONSOLE_CONFIG_FILE': console_config_content
    },
    'ports': [
        '8080:8080'
    ],
    'depends_on': [
        'redpanda'
    ]
}

compose = {
    'services': {
        SERVICE_NAME: redpanda_service,
        CONSOLE_SERVICE_NAME: redpanda_console_service
    },
    'volumes': {
        'redpanda': None
    }
}
