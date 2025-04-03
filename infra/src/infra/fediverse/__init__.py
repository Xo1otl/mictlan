from .env import *

DOMAIN = 'lemmy.mictlan.site'
SMTP_SERVER = 'mail.mictlan.site'
SMTP_DOMAIN = 'mictlan.site'


def compose():
    from .compose import services
    return {
        'services': services
    }
