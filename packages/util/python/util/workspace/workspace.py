from pathlib import Path
import pkgutil
from typing import Set, Dict
import importlib
import inspect
import sys
from typing import Set, Dict, List
import os
import glob
from typing import List


def findroot() -> str:
    """
    ワークスペースのルートを探し、見つかったルートを返す関数。
    'workspace.php' が存在するディレクトリをルートとする。

    :param start_dir: 開始地点となるディレクトリのパス
    :return: ワークスペースのルートディレクトリのパス
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(current_dir, 'workspace.php')):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("workspace.php が見つかりませんでした。")
        current_dir = parent_dir


root_dir = findroot()


# ワークスペースルートから指定されたパターンにマッチするファイルを絶対パスで取得する関数
def globpaths(pattern: str) -> List[str]:
    """
    グローバル変数 root_dir を使用して、指定されたパターンにマッチするファイルの絶対パスを返す。

    :param pattern: ワークスペースルートからの相対パスで指定するファイル検索パターン（glob形式）
    :return: マッチしたファイルの絶対パスのリスト
    """
    search_pattern = os.path.join(root_dir, pattern)
    matching_files = glob.glob(search_pattern)
    return [os.path.abspath(file_path) for file_path in matching_files]


# 上記の関数を利用して、相対パスを取得する関数
def globrelpaths(base: str, pattern: str) -> List[str]:
    """
    グローバル変数 root_dir を使用して、指定パターンにマッチするファイルのリストを
    開始ファイルの場所からの最短相対パスで返す。

    :param base: 開始地点となるファイルのパス（通常は __file__）
    :param pattern: ワークスペースルートからの相対パスで指定するファイル検索パターン（glob形式）
    :return: マッチしたファイルへの相対パスのリスト
    """
    # 開始ファイルのディレクトリを取得
    start_dir = os.path.dirname(os.path.abspath(base))

    # ワークスペースルートからパターンにマッチする絶対パスのリストを取得
    absolute_paths = globpaths(pattern)

    # 開始ディレクトリから各ファイルへの最短相対パスを計算
    relative_paths = [os.path.relpath(file_path, start_dir)
                      for file_path in absolute_paths]

    return relative_paths


def relpath(base: str, target: str) -> str:
    """
    グローバル変数 root_dir を使用して、指定されたファイルまたはフォルダへの最短相対パスを
    開始ファイルの場所から返す。

    :param base: 開始地点となるファイルのパス（通常は __file__）
    :param target: ワークスペースルートからの相対パスで指定するファイルまたはフォルダ
    :return: 開始地点からターゲットへの最短相対パス
    """
    # 開始ファイルのディレクトリを取得
    start_dir = os.path.dirname(os.path.abspath(base))

    # ターゲットの絶対パスを取得
    target_abs_path = os.path.abspath(os.path.join(root_dir, target))

    # 開始ディレクトリからターゲットへの相対パスを計算
    relative_path = os.path.relpath(target_abs_path, start_dir)

    return relative_path


def read(path: str) -> str:
    file_path = os.path.join(root_dir, path)
    with open(file_path, 'r') as f:
        return f.read()
