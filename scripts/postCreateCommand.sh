#!/bin/bash

# fishのpathは適用されないのでフルパスで実行
# スクリプト内で使う変数を定義
WORKSPACE_DIR="/workspaces/mictlan"
VENV_DIR="$WORKSPACE_DIR/.venv"

(
  # Bunパッケージのインストール
  (/home/vscode/.bun/bin/bun install --cwd $WORKSPACE_DIR) &

  # Python仮想環境のセットアップと依存関係のインストール
  (
    # .venvディレクトリが存在しない場合のみ作成
    [ ! -d "$VENV_DIR" ] && /home/vscode/.local/bin/uv venv --directory $WORKSPACE_DIR

    # 仮想環境を有効化
    . "$VENV_DIR/bin/activate"

    # GPUを検出した場合、--extra gpuフラグを変数にセット
    extra_args=""
    if nvidia-smi &> /dev/null; then
      echo "GPU detected. Installing with 'gpu' extra."
      extra_args="--extra gpu"
    fi
    
    # 依存関係を同期 (GPUフラグがあればそれを含める)
    /home/vscode/.local/bin/uv sync --directory $WORKSPACE_DIR $extra_args
  ) &

  # Git LFSの設定
  (
    git lfs install --skip-repo && \
    git -C $WORKSPACE_DIR lfs pull && \
    git -C $WORKSPACE_DIR config pull.rebase true && \
    git -C $WORKSPACE_DIR remote set-url origin git@github:Xo1otl/mictlan.git
  ) &
  
  # すべてのバックグラウンドジョブの終了を待つ
  wait
)

echo "post create command has completed."
