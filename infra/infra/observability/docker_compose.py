PORT = "3000"

# prometheus使うとnode exporter一緒に使う
docker_compose = {
    'services': {
        'grafana': {
            'image': 'grafana/grafana:latest',
            'container_name': 'grafana',
            'ports': [
                f'{PORT}:3000'
            ],
        }
    },
    'volumes': {
        'grafana_storage': {}
    }
}
