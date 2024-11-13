from infra import llm
from util import workspace

pipelines = {
    'image': 'ghcr.io/open-webui/pipelines:main',
    'container_name': 'pipelines',
    'ports': ['9099:9099'],
    'volumes': [
        f'{workspace.relpath(__file__, "apps/chathub/chathub/pipelines")}:/app/pipelines',
        f'{workspace.relpath(__file__, "apps/chathub/chathub/chat")}:/app/chat'
    ],
}

docker_compose = {
    'services': {
        'chat-pipelines': pipelines,
        'chat': {
            'image': 'ghcr.io/open-webui/open-webui:main',
            'container_name': 'open-webui',
            'ports': ['3000:8080'],
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
