from .env import *

CONF_FILENAME = 'configuration.yml'
PORT = 9091
DBUSER = 'authelia'
DBNAME = 'authelia'


def compose():
    from .authelia import authelia_service
    return {
        'services': {
            'authelia': authelia_service
        }
    }
