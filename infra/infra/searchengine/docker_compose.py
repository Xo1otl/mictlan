from . import *

docker_compose = {
    'services': {
        'meilisearch': {
            'image': 'getmeili/meilisearch:latest',
            'ports': ['7700:7700'],
            'environment': [
                f'MEILI_MASTER_KEY={MEILI_MASTER_KEY}',
            ],
            'volumes': [
                './meili-data:/meili_data',
            ]
        },
    }
}
