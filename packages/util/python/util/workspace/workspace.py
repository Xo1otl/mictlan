import os
import glob
from typing import List


def find_workspace_root(start_dir: str) -> str:
    """
    ワークスペースのルートを探し、見つかったルートを返す関数。
    'workspace.php' が存在するディレクトリをルートとする。

    :param start_dir: 開始地点となるディレクトリのパス
    :return: ワークスペースのルートディレクトリのパス
    """
    current_dir = start_dir
    while True:
        if os.path.isfile(os.path.join(current_dir, 'workspace.php')):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("workspace.php が見つかりませんでした。")
        current_dir = parent_dir


root_dir = find_workspace_root(os.path.dirname(os.path.abspath(__file__)))


def findrelpaths(base: str, pattern: str) -> List[str]:
    """
    グローバル変数 root_dir を使用して、指定パターンにマッチするファイルのリストを
    開始ファイルの場所からの最短相対パスで返す。

    :param base: 開始地点となるファイルのパス（通常は __file__）
    :param pattern: ワークスペースルートからの相対パスで指定するファイル検索パターン（glob形式）
    :return: マッチしたファイルへの相対パスのリスト
    """
    # 開始ファイルのディレクトリを取得
    start_dir = os.path.dirname(os.path.abspath(base))

    # ワークスペースルートからパターンにマッチするファイルを探す
    search_pattern = os.path.join(root_dir, pattern)
    matching_files = glob.glob(search_pattern)

    # 開始ディレクトリから各ファイルへの最短相対パスを計算
    relative_paths = [os.path.relpath(file_path, start_dir)
                      for file_path in matching_files]

    return relative_paths
