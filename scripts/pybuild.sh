#!/bin/bash

# プロジェクトルートを見つける関数
find_project_root() {
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
}

# PYTHONPATHを設定する関数
set_pythonpath() {
  export PYTHONPATH="$project_root/build:${PYTHONPATH}"
}

# 指定したパターンのPythonファイルを実行する関数
execute_python_files() {
  find_project_root
  set_pythonpath
  local pattern="$1"
  find "$project_root/build" -type f -name "$pattern" | while read -r file; do
    dir=$(dirname "$file")
    echo "Executing $file in $dir..."
    (cd "$dir" && python "$(basename "$file")")
  done
}
