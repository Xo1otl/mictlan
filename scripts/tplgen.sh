#!/bin/bash

# スクリプトのあるディレクトリを取得
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 共通スクリプトを読み込み
source "$DIR/buildcommon.sh"

# プロジェクトルートを見つける
find_project_root

# PYTHONPATHを設定
set_pythonpath

# *.tpl.py ファイルを実行
execute_python_files "*.tpl.py"
