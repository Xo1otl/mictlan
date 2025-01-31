from . import *

lldap_service = {
    "image": "lldap/lldap:stable",
    "ports": [f"{WEBUI_PORT}:17170"],
    "volumes": ["lldap_data:/data"],
    "environment": {
        "LLDAP_JWT_SECRET": LLDAP_JWT_SECRET,
        "LLDAP_KEY_SEED": LLDAP_KEY_SEED,
        "LLDAP_LDAP_BASE_DN": LLDAP_LDAP_BASE_DN,
        "LLDAP_LDAP_USER_PASS": LLDAP_LDAP_USER_PASS,
    },
}
