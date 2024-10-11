from infra import proxy
from . import *

# TODO: certbotでrenewするコマンドをcronする処理をentrypointに追加する方法考える
docker_compose = {
    "services": {
        "mailserver": {
            "image": "ghcr.io/docker-mailserver/docker-mailserver:latest",
            "container_name": CONTAINER_NAME,
            "hostname": HOSTNAME,
            "ports": [
                # SMTP (explicit TLS => STARTTLS, Authentication is DISABLED => use port 465/587 instead)
                "25:25",
                "143:143",  # IMAP4 (explicit TLS => STARTTLS)
                "465:465",  # ESMTP (implicit TLS)
                "587:587",  # ESMTP (explicit TLS => STARTTLS)
                "993:993",  # IMAP4 (implicit TLS)
            ],
            "volumes": [
                "./docker-data/dms/mail-data/:/var/mail/",
                "./docker-data/dms/mail-state/:/var/mail-state/",
                "./docker-data/dms/mail-logs/:/var/log/mail/",
                "./docker-data/dms/config/:/tmp/docker-mailserver/",
                f"{proxy.CERT_DIR}:/etc/letsencrypt:ro",
                "/etc/localtime:/etc/localtime:ro"
            ],
            "environment": [
                "ENABLE_RSPAMD=1",
                "SSL_TYPE=letsencrypt",
            ],
            'restart': 'no'
        }
    }
}
