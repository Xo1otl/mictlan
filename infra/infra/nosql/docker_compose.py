from . import *

docker_compose = {
    'services': {
        CONTAINER_NAME: {
            'image': 'mongo:latest',
            'container_name': CONTAINER_NAME,
            'ports': [f'{PORT}:{PORT}'],
        }
    }
}
