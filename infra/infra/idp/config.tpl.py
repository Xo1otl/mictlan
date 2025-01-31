import os
import yaml
from infra import idp, ldap

config = {
    "server": {
        "address": f'tcp://0.0.0.0:{idp.PORT}',
    },
    "log": {
        "level": "debug"
    },
    "totp": {
        "issuer": "auth.mictlan.site",
        "period": 30,
        "skew": 1,
    },
    "authentication_backend": {
        "password_reset": {
            "disable": False,
        },
        "ldap": {
            "address": f"ldap://lldap:{ldap.LDAP_PORT}",
            "base_dn": ldap.LLDAP_LDAP_BASE_DN,
            "user": ldap.ADMIN_USER,
            "password": ldap.LLDAP_LDAP_USER_PASS,
            "users_filter": '(&({username_attribute}={input})(objectClass=person))',
            "groups_filter": '(&(member={dn})(objectClass=groupOfNames))',
        }
    },
    "access_control": {
        "default_policy": "deny",
        "rules": [
            {
                "domain": "auth.mictlan.site",
                "policy": "bypass"
            }
        ]
    },
    "identity_validation": {
        "reset_password": {
            "jwt_secret": idp.JWT_SECRET,
        }
    },
    "session": {
        "name": "authelia_session",
        "same_site": "lax",
        "secret": idp.SESSION_SECRET,
        "cookies": [
            {
                "domain": "mictlan.site",
                "authelia_url": "https://auth.mictlan.site",
                "default_redirection_url": "https://blog.mictlan.site",
                "name": "authelia_session",
            }
        ],
        "redis": {
            "host": "redis",
            "port": 6379,
            "database_index": 3,
        }
    },
    "storage": {
        "encryption_key": idp.STORAGE_ENCRYPTION_KEY,
        "postgres": {
            "address": "postgres:5432",
            "database": "authelia",
            "schema": "public",
            "username": "authelia",
            "password": idp.POSTGRES_PASSWORD,
        }
    },
    "notifier": {
        "disable_startup_check": True,
        "smtp": {
            "username": "xolotl@mictlan.site",
            "password": idp.SMTP_PASSWORD,
            "address": "smtp://mail.mictlan.site:587",
            "sender": "xolotl@mictlan.site",
            "subject": "[Authelia] {title}",
            "startup_check_address": "test@authelia.com",
        }
    }
}

# 実行しているコンテキストによらずtplと同じ場所に出力
target = os.path.join(os.path.dirname(__file__), idp.CONF_FILENAME)

with open(target, 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

print(f"[idp] configuration has been written to {target}.")
