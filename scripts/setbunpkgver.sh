#!/bin/bash

# package.jsonからdependenciesとdevDependenciesを抽出
dependencies=$(jq -r '.dependencies | keys[]' package.json)
devDependencies=$(jq -r '.devDependencies | keys[]' package.json)

# dependenciesの更新
for dep in $dependencies
do
  if [ "$dep" != "pkg" ]; then  # workspaceのpkgは除外
    bun install --save $dep@latest
  fi
done

# devDependenciesの更新
for dep in $devDependencies
do
  bun install --save-dev $dep@latest
done

# peerDependenciesの更新（存在する場合）
if jq -e '.peerDependencies' package.json > /dev/null; then
  peerDependencies=$(jq -r '.peerDependencies | keys[]' package.json)
  for dep in $peerDependencies
  do
    bun install --save-peer $dep@latest
  done
fi

echo "All dependencies have been updated to their latest versions."