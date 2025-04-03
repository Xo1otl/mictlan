from infra.ai import llm
from workspace import path

pipelines = {
    'image': 'ghcr.io/open-webui/pipelines:main',
    'container_name': 'pipelines',
    'ports': ['9099:9099'],
    'volumes': [
        f'{path.Path("apps/chathub").rel2(path.Path(__file__).dir())}:/app/chathub',
        f'{path.Path("apps/chathub/repositories").rel2(path.Path(__file__).dir())}:/app/repositories',
        f'{path.Path("apps/chathub/chathub/pipelines").rel2(path.Path(__file__).dir())}:/app/pipelines',
    ],
}

compose = {
    'services': {
        'chat-pipelines': pipelines,
        'chat': {
            'image': 'ghcr.io/open-webui/open-webui:main',
            'container_name': 'open-webui',
            'ports': ['3080:8080'],
            'environment': [
                f'OLLAMA_BASE_URL={llm.OLLAMA_URL}'
            ],
            'volumes': ['open-webui:/app/backend/data'],
            'restart': 'always'
        }
    },
    'volumes': {
        'open-webui': {},
    }
}
