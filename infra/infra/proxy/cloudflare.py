from .env import *

cloudflare_service = {
    'image': 'cloudflare/cloudflared:latest',
    'command': f'tunnel --no-autoupdate run --token {CLOUDFLARE_TOEKN}',
}
