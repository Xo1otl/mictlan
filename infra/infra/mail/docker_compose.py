from infra import proxy
from . import *

# TODO: certbotでrenewするコマンドをcronする処理をentrypointに追加する方法考える
docker_compose = {
    "services": {
        "mailserver": {
            "image": "ghcr.io/docker-mailserver/docker-mailserver:latest",
            "container_name": CONTAINER_NAME,
            "hostname": FQDN,
            "ports": [
                "25:25", # SMTP (explicit TLS => STARTTLS)
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
                "RSPAMD_GREYLISTING=1",
                "ENABLE_OPENDKIM=0",
                "ENABLE_OPENDMARC=0",
                "ENABLE_POLICYD_SPF=0",
                "ENABLE_AMAVIS=0",
                "ENABLE_SPAMASSASSIN=0",
                "ENABLE_POSTGREY=0",
                "SSL_TYPE=letsencrypt",
                "DEFAULT_RELAY_HOST=[smtp.gmail.com]:587",
                f"RELAY_USER={RELAY_USER}",
                f"RELAY_PASSWORD={RELAY_PASSWORD}"
            ],
            'restart': 'no'  # このコンテナのせいでsshすらできなくなることがあるからnoにしておく、多分clamAVのせい
        }
    }
}
