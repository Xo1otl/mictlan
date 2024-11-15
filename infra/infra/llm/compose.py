from . import *

compose = {
    'services': {
        'ollama': {
            'image': 'ollama/ollama:latest',
            'container_name': OLLAMA_CONTAINER_NAME,
            'volumes': ['ollama:/root/.ollama'],
            'ports': [f'{PORT}:{PORT}'],
            'environment': [
                'OLLAMA_NUM_PARALLEL=16',  # デフォルトだと4しか無い
            ],
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
