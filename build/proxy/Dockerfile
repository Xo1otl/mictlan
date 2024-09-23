FROM nginx:alpine

# Install Certbot
RUN apk add certbot certbot-nginx

# Add Cronjob
RUN echo "0	0	*	*	*	/usr/bin/certbot renew --quiet" >> /etc/crontabs/root