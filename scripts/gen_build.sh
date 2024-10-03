#!/bin/bash

# プロジェクトルートの絶対パス
project_root="/workspaces/mictlan"

# ../build ディレクトリ内のすべての *.gen.py ファイルを再帰的に探して実行
find "$project_root/build" -type f -name "*.gen.py" | while read -r file; do
  # ファイルが存在するディレクトリに移動
  dir=$(dirname "$file")
  echo "Executing $file in $dir..."
  
  # 現在のディレクトリを保存し、処理後に戻る
  (cd "$dir" && python "$(basename "$file")")
done
