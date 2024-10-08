import vpn
from proxy import *
from .env import *

# Configuration as a Python object
docker_compose = {
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
