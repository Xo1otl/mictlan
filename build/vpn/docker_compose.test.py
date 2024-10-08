import yaml
# テストファイルでは相対インポートはできないのでモジュール名を指定してimportを行う
from vpn import *

# docker composeが正しく書けているかについてコンポーネントテストを行う
# シンプルにファイル出力してから目視で検証する
with open('docker-compose.yaml', 'w') as file:
    yaml.dump(docker_compose, file, default_flow_style=False)
