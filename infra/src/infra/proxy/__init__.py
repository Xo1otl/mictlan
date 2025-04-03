from .env import *

CONF_FILENAME = 'nginx.conf'


def compose():
    from .nginx import nginx_service
    from .cloudflare import cloudflare_service
    return {
        'services': {
            'nginx': nginx_service,
            'cloudflared': cloudflare_service
        }
    }
