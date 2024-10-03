import vpn
from proxy import *

compose = f"""\
services:
  nginx:
    build:
      dockerfile: Dockerfile
    environment:
      - VPN_HOST={vpn.HOST}
      - CERTBOT_EMAIL={CERTBOT_EMAIL}
    volumes:
      - ./entrypoint.sh:/entrypoint.sh
      - ../vpn/nginx.conf:/etc/nginx/conf.d/vpn.conf
    entrypoint: [ "/entrypoint.sh" ]
    ports:
      - "80:80/tcp"
      - "443:443/tcp"
    restart: unless-stopped
"""

print(compose)
