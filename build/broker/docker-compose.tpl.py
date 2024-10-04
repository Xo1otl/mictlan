import yaml

# Redpandaサービスの設定
redpanda_service = {
    'command': [
        'redpanda',
        'start',
        '--kafka-addr internal://0.0.0.0:9092,external://0.0.0.0:19092',
        '--advertise-kafka-addr internal://redpanda:9092,external://localhost:19092',
        '--pandaproxy-addr internal://0.0.0.0:8082,external://0.0.0.0:18082',
        '--advertise-pandaproxy-addr internal://redpanda:8082,external://localhost:18082',
        '--schema-registry-addr internal://0.0.0.0:8081,external://0.0.0.0:18081',
        '--rpc-addr redpanda:33145',
        '--advertise-rpc-addr redpanda:33145',
        '--mode dev-container',
        '--smp 1',
        '--default-log-level=info'
    ],
    'image': 'docker.redpanda.com/redpandadata/redpanda:latest',
    'container_name': 'redpanda',
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

console_config_content = '''kafka:
  brokers: ["redpanda:9092"]
  schemaRegistry:
    enabled: true
    urls: ["http://redpanda:8081"]
redpanda:
  adminApi:
    enabled: true
    urls: ["http://redpanda:9644"]'''

# Redpanda Consoleサービスの設定
redpanda_console_service = {
    'container_name': 'redpanda-console',
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

config = {
    'services': {
        'redpanda': redpanda_service,
        'redpanda-console': redpanda_console_service
    },
    'volumes': {
        'redpanda': None
    }
}

with open('docker-compose.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)
