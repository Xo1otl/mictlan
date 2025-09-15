#!/bin/bash
# エラーが発生した時点でスクリプトを終了する
set -e

# --- mise でツールをインストール ---
echo "Installing tools with mise..."
mise install

echo "Starting parallel setup for Bun, Python, and Git..."
(
  # --- Bun パッケージのインストール ---
  (
    echo "[Bun] Checking for existing node_modules..."
    if [ ! -d "node_modules" ]; then
      echo "[Bun] node_modules not found. Running bun install..."
      mise exec -- bun install
    else
      echo "[Bun] node_modules already exists. Skipping install."
    fi
  ) &

  # --- Python 仮想環境のセットアップ ---
  (
    echo "[Python/uv] Checking for pyproject.toml..."
    if [ -f "pyproject.toml" ]; then
      if [ ! -d ".venv" ]; then
        echo "[Python/uv] .venv not found. Creating virtual environment..."
        mise exec -- uv venv
      else
        echo "[Python/uv] .venv already exists."
      fi

      echo "[Python/uv] Syncing dependencies..."

      extra_args=""
      echo "[Python/uv] Checking for CUDA toolkit directory..."
      # /usr/local/cuda-* というパターンのディレクトリが存在するかどうかで判定
      if ls -d /usr/local/cuda-*/ >/dev/null 2>&1; then
        echo "[Python/uv] CUDA toolkit directory found. Installing with 'cuda' extra."
        extra_args="--extra cuda"
      else
        echo "[Python/uv] CUDA toolkit directory not found. Skipping 'cuda' extra."
      fi
      mise exec -- uv sync $extra_args
    else
      echo "[Python/uv] pyproject.toml not found. Skipping setup."
    fi
  ) &

  # --- Git の設定 ---
  (
    echo "[Git] Configuring LFS and remote..."
    git lfs install --skip-repo && \
    git lfs pull && \
    git config pull.rebase true && \
    git remote set-url origin git@github.com:Xo1otl/mictlan.git
  ) &

  # --- すべてのバックグラウンドジョブの終了を待つ ---
  wait
)

echo "Post create command has completed successfully."
