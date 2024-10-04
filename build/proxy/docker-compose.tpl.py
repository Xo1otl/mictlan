import yaml
import vpn
from proxy import *

# Configuration as a Python object
config = {
    'services': {
        'nginx': {
            'build': {
                'dockerfile': 'Dockerfile'
            },
            'environment': [
                f'VPN_HOST={vpn.HOST}',
                f'CERTBOT_EMAIL={CERTBOT_EMAIL}'
            ],
            'volumes': [
                './entrypoint.sh:/entrypoint.sh',
                f'../vpn/{vpn.NGINX_CONF_FILE}:/etc/nginx/conf.d/vpn.conf'
            ],
            'entrypoint': ["/entrypoint.sh"],
            'ports': [
                '80:80/tcp',
                '443:443/tcp'
            ],
            'restart': 'unless-stopped'
        }
    }
}

# Write the configuration to a docker-compose.yaml file
with open('docker-compose.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)
