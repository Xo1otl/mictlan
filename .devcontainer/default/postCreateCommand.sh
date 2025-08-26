#!/bin/bash

# fishのpathは適用されないのでフルパスで実行
(
  (/home/vscode/.bun/bin/bun install --cwd /workspaces/mictlan) &
  (/home/vscode/.local/bin/uv venv --directory /workspaces/mictlan && . /workspaces/mictlan/.venv/bin/activate && /home/vscode/.local/bin/uv sync --directory /workspaces/mictlan) &
  (
    # workspace managerを先にインストール
    /home/vscode/.local/bin/composer install --working-dir=/workspaces/mictlan/scripts/phpm &&
    # workspace managerを使用してworkspaceの依存関係をインストール
    cd /workspaces/mictlan && /workspaces/mictlan/scripts/phpm.sh install
  ) &
  (
    git lfs install --skip-repo &&
    git -C /workspaces/mictlan lfs pull &&
    git -C /workspaces/mictlan config pull.rebase true &&
    git -C /workspaces/mictlan remote set-url origin git@github:Xo1otl/mictlan.git
  ) &
  wait
)

echo "post create command has completed."
