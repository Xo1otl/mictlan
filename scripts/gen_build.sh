#!/bin/bash

# 現在のディレクトリから上位のディレクトリをたどってworkspace.phpを探す
current_dir=$(pwd)

while [ "$current_dir" != "/" ]; do
  if [ -f "$current_dir/workspace.php" ]; then
    project_root="$current_dir"
    break
  fi
  current_dir=$(dirname "$current_dir")
done

# workspace.phpが見つからない場合はエラーを表示して終了
if [ -z "$project_root" ]; then
  echo "workspace.php が見つかりませんでした。スクリプトを終了します。"
  exit 1
fi

echo "プロジェクトルート: $project_root"

# 既存のPYTHONPATHにbuildディレクトリを追加
export PYTHONPATH="$project_root/build:${PYTHONPATH}"

# ../build ディレクトリ内のすべての *.gen.py ファイルを探して実行
find "$project_root/build" -type f -name "*.gen.py" | while read -r file; do
  # ファイルが存在するディレクトリに移動
  dir=$(dirname "$file")
  echo "Executing $file in $dir..."
  
  # 現在のディレクトリを保存し、処理後に戻る
  (cd "$dir" && python "$(basename "$file")")
done
