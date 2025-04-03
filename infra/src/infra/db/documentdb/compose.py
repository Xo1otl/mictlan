from . import *

compose = {
    'services': {
        SERVICE_NAME: {
            'image': 'mongo:latest',
            'container_name': SERVICE_NAME,
            'ports': [f'{PORT}:{PORT}'],
        }
    }
}
