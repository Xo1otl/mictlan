#!/bin/bash
# 作業ディレクトリへ移動
cd /workspaces/mictlan/apps/nback

# ビルド実行
bun run build

# PIDファイルとログファイルのパスを定義
PID_FILE="./out/preview.pid"
LOG_FILE="./out/server.log"

# 既に実行中の preview プロセスがあれば停止する
if [ -f "$PID_FILE" ]; then
  old_pid=$(cat "$PID_FILE")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "Stopping previous preview process (PID: $old_pid)..."
    kill "$old_pid"
    # プロセスが終了するのを少し待つ
    sleep 1
    # 万が一プロセスが残っていたら強制終了
    if kill -0 "$old_pid" 2>/dev/null; then
      echo "Process did not terminate gracefully. Forcing kill..."
      kill -9 "$old_pid"
    fi
  else
    echo "PID file exists but process $old_pid is not running."
  fi
  rm "$PID_FILE"
fi

# preview プロセスをバックグラウンドで起動し、ログを出力
nohup bun run preview &> "$LOG_FILE" &
preview_pid=$!

# 新しい PID をハードコードしたファイル名で保存
echo "$preview_pid" > "$PID_FILE"
echo "Started preview process with PID: $preview_pid"
