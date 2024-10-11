import os
from infra import vpn
from infra import proxy

config = f"""\
server {{
    server_name {vpn.HOSTNAME};

    location / {{
        proxy_pass http://wg-easy:{vpn.WEBUI_PORT}/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }}
}}
"""

# 実行しているコンテキストによらずtplと同じ場所に出力
target = os.path.join(os.path.dirname(__file__), proxy.CONF_FILENAME)

with open(target, 'w') as file:
    file.write(config)

print(f"nginx.conf has been written to {target}.")
