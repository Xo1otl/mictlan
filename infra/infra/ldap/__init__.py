from .env import *

LDAP_PORT = 3890
WEBUI_PORT = 17170
PORTS = [LDAP_PORT, WEBUI_PORT]
LLDAP_LDAP_BASE_DN = "dc=mictlan,dc=site"


def compose():
    from .lldap import lldap_service
    return {
        'volumes': {
            'lldap_data': None
        },
        'services': {
            'lldap': lldap_service
        }
    }
