#!/bin/bash
# -----------------------------------------------------------------------------
# initializeCommand.sh: コンテナ構築前にホストマシンで実行される
#
# 主な役割:
# 1. Gitサブモジュールの初期化と更新
# 2. DockerfileのCOPYで使われるファイルの事前展開
# -----------------------------------------------------------------------------

# コマンドが失敗した場合、または未定義の変数を使用した場合にスクリプトを終了させる
set -euo pipefail

# スクリプトの開始位置から親ディレクトリを遡り、指定されたファイルを見つける
MARKER_FILE="mictlan.code-workspace"
SEARCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

while true; do
  # マーカーファイルが見つかったら、そこをプロジェクトルートとしてループを抜ける
  if [ -f "$SEARCH_DIR/$MARKER_FILE" ]; then
    PROJECT_ROOT="$SEARCH_DIR"
    break
  fi
  # ルートディレクトリ('/')まで遡っても見つからなければエラー終了（無限ループ防止）
  if [ "$SEARCH_DIR" = "/" ]; then
    echo "Error: Could not find workspace root containing '$MARKER_FILE'." >&2
    exit 1
  fi
  # 親ディレクトリに移動
  SEARCH_DIR=$(dirname "$SEARCH_DIR")
done

# 特定したプロジェクトのルートディレクトリに移動
cd "$PROJECT_ROOT"
echo "✅ Found workspace root and changed directory to: ${PROJECT_ROOT}"

# --- 初期化処理が既に実行済みか確認 ---
INITIALIZED_MARKER=".initialized"
if [ -f "$INITIALIZED_MARKER" ]; then
  echo "✅ Project already initialized. Skipping setup."
  exit 0
fi

# --- 1. Gitサブモジュールの初期化と更新 ---
echo "▶ Initializing and updating Git submodules..."
git submodule update --init --recursive
echo "✔ Submodules updated."

echo "▶ Checking out the 'main' branch for each submodule..."
git submodule foreach 'git checkout main || true'
echo "✔ Submodules are on the 'main' branch."

# --- 2. 機密情報ファイル (secrets.tar.gz) の展開 ---
# DockerfileのCOPYで使用するために、事前に展開しておく
SECRETS_FILE="secrets.tar.gz"

echo "▶ Checking for secrets archive ($SECRETS_FILE)..."
if [ -f "$SECRETS_FILE" ]; then
  echo "  Found ${SECRETS_FILE}. Extracting for Dockerfile COPY..."
  tar -xzvf "$SECRETS_FILE"
  
  echo "✔ Secrets have been successfully extracted."
else
  echo "  No ${SECRETS_FILE} found. Skipping extraction."
fi

# --- 初期化完了のマーカーファイルを作成 ---
echo "▶ Creating initialization marker file..."
touch "$INITIALIZED_MARKER"
echo "✔ Marker file created."

echo "✅ Initialization command finished successfully."
