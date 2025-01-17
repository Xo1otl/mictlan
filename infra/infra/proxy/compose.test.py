import yaml
import sys
# テストファイルでは相対インポートはできないのでモジュール名を指定してimportを行う
from infra import proxy

# docker composeが正しく書けているかについてコンポーネントテストを行う
# シンプルにコマンドラインに出力して目視で検証する
# テスト用のdocker-compose.yamlはnameが設定されてないので直で使うべきじゃない
yaml.dump(proxy.compose(), sys.stdout, default_flow_style=False)
