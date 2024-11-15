from . import *

compose = {
    'services': {
        'answer': {
            'container_name': 'answer',
            'build': {
                'dockerfile': 'Dockerfile',
            },
            'ports': [
                f'{PORT}:80',
            ],
            'volumes': [
                'answer-data:/data',
            ],
        },
    },
    'volumes': {
        'answer-data': None,
    },
}
