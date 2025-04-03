import yaml
import os

config_filename = "workspace.yaml"


def findroot() -> str:
    """
    ワークスペースのルートを探し、見つかったルートを返す関数。

    :param start_dir: 開始地点となるディレクトリのパス
    :return: ワークスペースのルートディレクトリのパス
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(current_dir, config_filename)):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(f"{config_filename} が見つかりませんでした。")
        current_dir = parent_dir


WORKSPACE_ROOT = findroot()
WORKSPACE_NAME = yaml.safe_load(
    open(os.path.join(WORKSPACE_ROOT, config_filename)))["name"]
