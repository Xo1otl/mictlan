from . import *

environment = {
    "PUBLIC_APP_URL": "https://auth.mictlan.site",
    "TRUST_PROXY": True,
    "MAXMIND_LICENSE_KEY": MAXMIND_LICENSE_KEY,
}

pocketid_service = {
    "image": "stonith404/pocket-id",
    "restart": "unless-stopped",
    "environment": environment,
    "volumes": [
        "pocketid_data:/app/backend/data"
    ],
    # Optional healthcheck
    "healthcheck": {
        "test": "curl -f http://localhost/health",
        "interval": "1m30s",
        "timeout": "5s",
        "retries": 2,
        "start_period": "10s"
    }
}

pocketid_volumes = {
    "pocketid_data": None
}
