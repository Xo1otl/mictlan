import yaml
from observability import *

config = {
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

with open('docker-compose.yaml', "w") as f:
    yaml.dump(config, f, default_flow_style=False)
