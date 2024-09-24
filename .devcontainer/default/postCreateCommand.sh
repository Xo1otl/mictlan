#!/bin/bash

# fishのpathは適用されないのでフルパスで実行
(
  (/home/vscode/.bun/bin/bun install --cwd /workspaces/mictlan) &
  (/home/vscode/.local/bin/poetry install --directory /workspaces/mictlan) &
  (
    # workspace managerを先にインストール
    /usr/local/bin/composer install --working-dir=/workspaces/mictlan/scripts/phpm &&
    # workspace managerを使用してworkspaceの依存関係をインストール
    cd /workspaces/mictlan && /workspaces/mictlan/scripts/phpm.sh install
  ) &
  (
    git lfs install --skip-repo &&
    git -C /workspaces/mictlan remote set-url origin git@github:Xo1otl/mictlan.git && 
    git -C /workspaces/mictlan lfs pull &&
    git -C /workspaces/mictlan config pull.rebase true
  ) &
  wait
)

echo "post create command has completed."