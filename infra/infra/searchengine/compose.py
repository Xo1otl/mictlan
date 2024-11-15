from . import *

compose = {
    'services': {
        'meilisearch': {
            'image': 'getmeili/meilisearch:latest',
            'ports': [f'{PORT}:{PORT}'],
            'environment': [
                f'MEILI_MASTER_KEY={MEILI_MASTER_KEY}',
            ],
            'volumes': [
                './meili-data:/meili_data',
            ]
        },
    }
}
