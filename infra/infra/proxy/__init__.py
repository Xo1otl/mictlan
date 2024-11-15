from .env import *

CONF_FILENAME = 'nginx.conf'


def compose():
    from .compose import compose
    return compose
