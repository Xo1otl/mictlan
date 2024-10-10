#!/bin/sh

# 動作テストの時は --test-cert をつける
certbot --nginx --non-interactive --agree-tos -m $CERTBOT_EMAIL -d $VPN_HOST --test-cert

# Dockerfileで設定したcronを起動
crond

# certbotがnginxを起動したのを止める
nginx -s stop

# 適切なオプションをつけてnginxを起動
exec nginx -g 'daemon off;'
