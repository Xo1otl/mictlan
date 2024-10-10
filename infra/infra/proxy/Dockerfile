FROM nginx:alpine

# CertbotはSSL証明書を取得するためのパッケージ
RUN apk add certbot certbot-nginx

# 毎日午前0時にSSL証明書を更新するcronジョブを追加
RUN echo "0	0	*	*	*	/usr/bin/certbot renew --quiet" >> /etc/crontabs/root