#!/bin/bash
# -----------------------------------------------------------------------------
# initializeCommand.sh: Devcontainerの初期化処理
#
# ホストマシン上でコンテナがビルドされる前に一度だけ実行されます。
# 主な役割:
# 1. Gitサブモジュールの初期化と更新
# 2. 機密情報ファイル (secrets.tar) の展開
# -----------------------------------------------------------------------------

# コマンドが失敗した場合、または未定義の変数を使用した場合にスクリプトを終了させる
set -euo pipefail

# スクリプトの場所を基準にプロジェクトルートを特定
# これにより、どのディレクトリから実行しても正しく動作する
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# プロジェクトのルートディレクトリに移動
cd "$PROJECT_ROOT"

# --- 1. Gitサブモジュールの初期化と更新 ---
echo "▶ Initializing and updating Git submodules..."
git submodule update --init --recursive
echo "✔ Submodules updated."

echo "▶ Checking out the 'main' branch for each submodule..."
git submodule foreach 'git checkout main'
echo "✔ Submodules are on the 'main' branch."

# --- 2. 機密情報ファイル (secrets.tar) の展開 ---
SECRETS_FILE="secrets.tar"

echo "▶ Checking for secrets archive ($SECRETS_FILE)..."
if [ -f "$SECRETS_FILE" ]; then
  echo "  Found ${SECRETS_FILE}. Extracting..."
  tar -xvf "$SECRETS_FILE"
  
  echo "✔ Secrets have been successfully extracted."
else
  echo "  No ${SECRETS_FILE} found. Skipping extraction."
fi

echo "✅ Initialization command finished successfully."
