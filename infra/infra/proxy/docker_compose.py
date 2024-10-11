from infra import vpn
from infra import mail
from util import workspace
from . import *
import os

# conf_filesにすべてのnginx.confファイルの相対パスを取得
conf_files = workspace.globrelpaths(__file__, 'infra/infra/**/nginx.conf')

# volumesを初期化し、entrypoint.shを追加
volumes = [
    './entrypoint.sh:/entrypoint.sh',
    f'{CERT_DIR}:/etc/letsencrypt'
]

for conf_file in conf_files:
    # フォルダ名を取得
    folder = os.path.basename(os.path.dirname(conf_file))
    # ターゲットパスを設定
    target_path = f'/etc/nginx/conf.d/{folder}.conf'
    # マッピングをvolumesに追加
    volumes.append(f'{conf_file}:{target_path}')

# docker_compose設定を作成
docker_compose = {
    'services': {
        'nginx': {
            'build': {
                'dockerfile': 'Dockerfile'
            },
            'environment': [
                f'VPN_HOST={vpn.HOSTNAME}',
                f'MAIL_HOST={mail.HOSTNAME}',
                f'CERTBOT_EMAIL={CERTBOT_EMAIL}'
            ],
            'volumes': volumes,
            'entrypoint': ["/entrypoint.sh"],
            'ports': [
                '80:80/tcp',
                '443:443/tcp'
            ],
            'restart': 'unless-stopped'
        }
    }
}
