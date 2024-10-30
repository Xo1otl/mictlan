import yaml
import os
from infra import vpn
from infra import searchengine
from infra import rdb
from infra import proxy
from infra import observability
from infra import broker
from infra import mail
from infra import ossekai
from infra import nosql
from infra import zaiko
from infra import asyncsns


# docker composeはまとめてやりたい処理とかが考えられるからここでまとめて生成する
# すべての設定にまとめてnetworksを追加するなどの共通処理が考えられる
def generate_docker_compose():
    modules = [
        vpn, searchengine, rdb, proxy, observability, broker, mail, ossekai, nosql, zaiko, asyncsns
    ]

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
            yaml.dump(module.docker_compose(), file, default_flow_style=False)

        print(
            f'[ecosystem] docker-compose.yaml has been written to: {output_file}'
        )

        # 生成したdocker-compose.yamlのパスをincludeリストに追加
        # 相対パスで生成するのでそのままほかのpcに持ってって使える
        includes.append(os.path.relpath(
            output_file, os.path.dirname(__file__)))

    # mictlan用のdocker compose構造を作成
    docker_compose = {
        'name': 'mictlan',
        'include': includes
    }

    # docker-compose.yamlを作成
    output_path = os.path.join(
        os.path.dirname(__file__), 'docker-compose.yaml')

    # mictlanのdocker-compose.yamlを書き込み
    with open(output_path, 'w') as file:
        yaml.dump(docker_compose, file, default_flow_style=False)

    print(
        f'[ecosystem] docker-compose.yaml has been written to: {output_path}'
    )
