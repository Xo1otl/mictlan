import yaml
from vpn import *

config = {
    'services': {
        'wg-easy': {
            'image': 'ghcr.io/wg-easy/wg-easy:latest',
            'environment': {
                'WG_HOST': HOST,
                'PASSWORD_HASH': WEBUI_PASSWORD_HASH,
                'WG_PORT': UDP_PORT,
                'PORT': WEBUI_PORT,
                'WG_ALLOWED_IPS': ALLOWED_IPS,
                'WG_PERSISTENT_KEEPALIVE': 25,
                'WG_DEFAULT_DNS': ''  # 空でDNSを使用しない
            },
            'ports': [
                f'{UDP_PORT}:{UDP_PORT}/udp'  # 既存の変数を使用
            ],
            'cap_add': [
                'NET_ADMIN',
                'SYS_MODULE'
            ],
            'sysctls': [
                'net.ipv4.conf.all.src_valid_mark=1',
                'net.ipv4.ip_forward=1'
            ],
            'restart': 'unless-stopped'
        }
    }
}

# YAMLファイルに出力します
with open('docker-compose.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)
