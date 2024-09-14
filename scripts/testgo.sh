#!/bin/bash

set -euo pipefail

# ファイルが存在するか確認
if [ ! -f "go.work" ]; then
    echo "Error: go.work file not found in the current directory." >&2
    exit 1
fi

# go.workファイルから'use'行を抽出し、括弧内のパスを取得
packages=$(sed -n '/^use/,/)/p' go.work | grep -v '^use' | sed 's/[()]//g' | tr -d '\t' | tr -d ' ')

# 各パッケージに対してテストを並列実行
for package in $packages; do
    if [ -d "$package" ]; then
        (
            echo "Running tests for $package"
            cd "$package" && go test ./... $@ || exit 1
        ) &
    else
        echo "Warning: Directory $package not found. Skipping..." >&2
    fi
done

# すべてのバックグラウンドジョブが終了するのを待つ
wait

echo "All tests completed."