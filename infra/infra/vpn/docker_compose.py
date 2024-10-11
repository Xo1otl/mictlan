from . import *

docker_compose = {
    'services': {
        'wg-easy': {
            'image': 'ghcr.io/wg-easy/wg-easy:latest',
            'environment': {
                'WG_HOST': FQDN,
                'PASSWORD_HASH': WEBUI_PASSWORD_HASH,
                'WG_PORT': UDP_PORT,
                'PORT': WEBUI_PORT,
                'WG_ALLOWED_IPS': ALLOWED_IPS,
                'WG_PERSISTENT_KEEPALIVE': 25,
                'WG_DEFAULT_DNS': ''  # 空でDNSを使用しない
            },
            'ports': [
                f'{UDP_PORT}:{UDP_PORT}/udp'
            ],
            'cap_add': [
                'NET_ADMIN',
                'SYS_MODULE'
            ],
            'sysctls': [
                'net.ipv4.conf.all.src_valid_mark=1',
                'net.ipv4.ip_forward=1'
            ],
            'restart': 'unless-stopped',
            'volumes': [
                '~/.wg-easy:/etc/wireguard'
            ]
        }
    }
}
