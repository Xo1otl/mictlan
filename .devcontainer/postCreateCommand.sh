#!/bin/bash

(
  (/home/vscode/.bun/bin/bun install --cwd /workspaces/mictlan/js) &
  (/home/vscode/.local/bin/poetry install --directory /workspaces/mictlan/python --no-root) &
  (
    git lfs install &&
    git -C /workspaces/mictlan remote set-url origin git@mictlan:Xo1otl/mictlan.git && 
    git -C /workspaces/mictlan lfs pull
  ) &
  wait
)

echo "post create command has completed."