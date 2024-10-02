#!/bin/sh

# conf.dに関係ないファイル配置すると読み込まれてエラーになるので場所変えてる
envsubst '$WG_HOST $WG_WEBUI_PORT' < /etc/nginx/templates/wg-easy.conf.template > /etc/nginx/conf.d/wg-easy.conf

# 動作テストの時は --test-cert をつける
certbot --nginx --non-interactive --agree-tos -m $CERTBOT_EMAIL -d $WG_HOST --test-cert

# Dockerfileで設定したcronを起動
crond

# certbotがnginxを起動したのを止める
nginx -s stop

# 適切なオプションをつけてnginxを起動
exec nginx -g 'daemon off;'
