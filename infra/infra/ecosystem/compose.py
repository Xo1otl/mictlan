import yaml
import os
from infra import vpn
from infra.db import searchengine
from infra.db import rdb
from infra import proxy
from infra import observability
from infra import broker
from infra import mail
from infra import ossekai
from infra.db import documentdb
from infra import zaiko
from infra import asyncsns
from infra.ai import llm
from infra.ai import imagegen
from infra import chat
from infra.db import multimodaldb
from infra.db import kvs
from infra import knowledgebase
from infra import akinator
from infra import fediverse


def check_port_conflicts(modules):
    port_usage = {}

    has_conflicts = False

    for module in modules:
        ports = []

        if hasattr(module, 'PORTS'):
            if isinstance(module.PORTS, (list, tuple)):
                ports.extend(module.PORTS)
            elif isinstance(module.PORTS, dict):
                ports.extend(module.PORTS.values())
        elif hasattr(module, 'PORT'):
            ports.append(module.PORT)
        else:
            continue

        for port in ports:
            if port in port_usage:
                has_conflicts = True
                print(f'[ecosystem] Port conflict detected: Port {port} is used by both '
                      f'{port_usage[port].__name__} and {module.__name__}')
            else:
                port_usage[port] = module

    if not has_conflicts:
        print('[ecosystem] No port conflicts detected')

    return not has_conflicts


# docker composeはまとめてやりたい処理とかが考えられるからここでまとめて生成する
# すべての設定にまとめてnetworksを追加するなどの共通処理が考えられる
def gen_compose():
    modules = [
        vpn, searchengine, rdb, proxy, observability, broker, mail, ossekai, documentdb, zaiko, asyncsns, llm, chat, multimodaldb, imagegen, kvs, knowledgebase, akinator, fediverse
    ]

    check_port_conflicts(modules)

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
            yaml.dump(module.compose(), file, default_flow_style=False)

        print(
            f'[ecosystem] docker-compose.yaml has been written to: {output_file}'
        )

        # 生成したdocker-compose.yamlのパスをincludeリストに追加
        # 相対パスで生成するのでそのままほかのpcに持ってって使える
        includes.append(os.path.relpath(
            output_file, os.path.dirname(__file__)))

    # mictlan用のdocker compose構造を作成
    compose = {
        'name': 'mictlan',
        'include': includes
    }

    # docker-compose.yamlを作成
    output_path = os.path.join(
        os.path.dirname(__file__), 'docker-compose.yaml')

    # mictlanのdocker-compose.yamlを書き込み
    with open(output_path, 'w') as file:
        yaml.dump(compose, file, default_flow_style=False)

    print(
        f'[ecosystem] docker-compose.yaml has been written to: {output_path}'
    )
