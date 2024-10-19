from . import *

ZIP_URL = "https://github.com/meln5674/grafana-mongodb-community-plugin/releases/download/v0.2.0%2Brc4/meln5674-mongodb-community.zip"

# prometheus使うとnode exporter一緒に使う
docker_compose = {
    'services': {
        'grafana': {
            'image': 'grafana/grafana-oss:latest',
            'container_name': 'grafana',
            # "environment": [
            #     "GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=meln5674-mongodb-community",
            #     f"GF_INSTALL_PLUGINS={ZIP_URL};meln5674-mongodb-community"
            # ],
            'ports': [
                f'{PORT}:3000'
            ],
        }
    },
    'volumes': {
        'grafana_storage': {}
    }
}
