import yaml
import os
import vpn
import searchengine
import relationaldb
import proxy
import observability
import broker
import mail

modules = [vpn, searchengine, relationaldb, proxy, observability, broker, mail]

# includeするパスを保持するリスト
includes = []

# 各モジュールのディレクトリにdocker-compose.yamlを保存
for module in modules:
    # モジュールのファイルパスを取得
    module_dir = os.path.dirname(str(module.__file__))

    # yamlファイルのパスを作成
    output_file = os.path.join(module_dir, 'docker-compose.yaml')

    # yamlファイルに書き込み
    with open(output_file, 'w') as file:
        yaml.dump(module.docker_compose, file, default_flow_style=False)

    print(f'docker-compose.yaml has been written to: {output_file}')

    # 生成したdocker-compose.yamlのパスをincludeリストに追加
    # 相対パスで生成するのでそのままほかのpcに持ってって使える
    includes.append(os.path.relpath(output_file, os.path.dirname(__file__)))

# mictlan用のdocker compose構造を作成
mictlan_compose = {
    'name': 'mictlan',
    'include': includes
}

# プロジェクトルートにmictlan用のdocker-compose.yamlを作成
mictlan_file = os.path.join(os.getcwd(), 'docker-compose.yaml')

# mictlanのdocker-compose.yamlを書き込み
with open(mictlan_file, 'w') as file:
    yaml.dump(mictlan_compose, file, default_flow_style=False)

print(f'mictlan docker-compose.yaml has been written to: {mictlan_file}')
