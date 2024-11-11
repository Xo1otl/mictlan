from infra import llm

docker_compose = {
    'services': {
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
        'open-webui': {}
    }
}
