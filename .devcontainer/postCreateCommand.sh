#!/bin/bash

(
  (/home/vscode/.bun/bin/bun install --cwd /workspaces/mictlan/js) &
  (/home/vscode/.local/bin/poetry install --directory /workspaces/mictlan/python --no-root) &
  (
    git lfs install --skip-repo &&
    git -C /workspaces/mictlan remote set-url origin git@mictlan:Xo1otl/mictlan.git && 
    git -C /workspaces/mictlan lfs pull &&
    git -C /workspaces/mictlan config pull.rebase true
  ) &
  wait
)

echo "post create command has completed."