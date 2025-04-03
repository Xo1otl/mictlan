from . import *

compose = {
    'services': {
        SERVICE_NAME: {
            'image': 'surrealdb/surrealdb:latest',
            'ports': [f'{PORT}:{PORT}'],
            'command': 'start'
        }
    }
}
