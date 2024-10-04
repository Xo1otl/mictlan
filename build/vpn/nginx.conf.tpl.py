from vpn import *

config = f"""\
server {{
    server_name {HOST};

    location / {{
        proxy_pass http://wg-easy:{WEBUI_PORT}/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }}
}}
"""

with open(NGINX_CONF_FILE, 'w') as file:
    file.write(config)
