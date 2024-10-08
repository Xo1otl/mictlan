#!/bin/bash

# スクリプトのあるディレクトリを取得
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 共通スクリプトを読み込み
source "$DIR/pybuild.sh"

# *.tpl.py ファイルを実行
execute_python_files "*.tpl.py"
