from . import *

docker_compose = {
    'services': {
        'localstack': {
            'container_name': 'localstack',
            'image': 'localstack/localstack-pro',
            'ports': [
                '4566:4566',
                '4510-4559:4510-4559',
                '443:443',
            ],
            'environment': [
                f'LOCALSTACK_AUTH_TOKEN={LOCALSTACK_AUTH_TOKEN}',
                'DEBUG=0',
                'PERSISTENCE=0',
            ],
            'volumes': [
                './volume:/var/lib/localstack',
                '/var/run/docker.sock:/var/run/docker.sock',
            ],
        }
    }
}
