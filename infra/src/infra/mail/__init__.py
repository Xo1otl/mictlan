from infra import ecosystem
from .env import *

FQDN = f'mail.{ecosystem.DOMAIN}'
CONTAINER_NAME = 'mailserver'
SMTP_PORT = 25
ESMTP_PORT = 587
IMAP4_PORT = 993
PORTS = [SMTP_PORT, ESMTP_PORT, IMAP4_PORT]


def compose():
    from .compose import compose
    return compose
