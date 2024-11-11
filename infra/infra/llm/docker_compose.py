from . import *

docker_compose = {
    'services': {
        'ollama': {
            'image': 'ollama/ollama:latest',
            'container_name': OLLAMA_CONTAINER_NAME,
            'volumes': ['ollama:/root/.ollama'],
            'ports': [f'{PORT}:{PORT}'],
            'deploy': {
                'resources': {
                    'reservations': {
                        'devices': [{
                            'driver': 'nvidia',
                            'capabilities': ['gpu'],
                            'count': 'all'
                        }]
                    }
                }
            }
        },
    },
    'volumes': {
        'ollama': None
    }
}
