#!/bin/sh

# Issue or update certificate and configure Nginx for HOSTS
# certbotはすでに証明書がある場合はスキップしてconfのみ更新してくれる
certbot --nginx --non-interactive --agree-tos -m "$CERTBOT_EMAIL" -d "$VPN_HOST"
certbot --nginx --non-interactive --agree-tos -m "$CERTBOT_EMAIL" -d "$MAIL_HOST"

# Start cron daemon and manage Nginx
crond
nginx -s stop
exec nginx -g 'daemon off;'
