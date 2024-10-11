#!/bin/sh

# Check for existing certificates
VPN_CERT_EXISTS=$(certbot certificates 2>/dev/null | grep -q "Domains: $VPN_HOST" && echo "yes" || echo "no")
MAIL_CERT_EXISTS=$(certbot certificates 2>/dev/null | grep -q "Domains: $MAIL_HOST" && echo "yes" || echo "no")

if [ "$VPN_CERT_EXISTS" = "yes" ] || [ "$MAIL_CERT_EXISTS" = "yes" ]; then
  echo "At least one existing certificate found. Attempting to renew..."
  certbot renew
else
  echo "No existing certificates found. Issuing new certificates..."
fi

# Issue certificate for VPN_HOST if it doesn't exist
if [ "$VPN_CERT_EXISTS" = "no" ]; then
  certbot --nginx --non-interactive --agree-tos -m "$CERTBOT_EMAIL" -d "$VPN_HOST" --test-cert
fi

# Issue certificate for MAIL_HOST if it doesn't exist
if [ "$MAIL_CERT_EXISTS" = "no" ]; then
  certbot certonly --nginx --non-interactive --agree-tos -m "$CERTBOT_EMAIL" -d "$MAIL_HOST" --test-cert
fi

# Start cron daemon and manage nginx as before
crond
nginx -s stop
exec nginx -g 'daemon off;'
