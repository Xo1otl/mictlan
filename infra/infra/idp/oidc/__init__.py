from .env import *

def compose():
    from .pocketid import pocketid_service, pocketid_volumes
    return {
        'services': {
            'oidc': pocketid_service
        },
        'volumes': pocketid_volumes
    }
