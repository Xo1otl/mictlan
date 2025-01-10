#!/usr/bin/env python
from util import workspace
from os import path
import subprocess
from datetime import datetime
import tempfile

# リモートサーバーの設定
ssh_remote = "koemade"

# ターゲットフォルダのパス
target = path.join(workspace.root_dir, "apps/koemade")

# 現在の日付を取得してファイル名に追加
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
archive_name = f"koemade_{current_date}.tar.gz"

# 一時ディレクトリにアーカイブを作成
with tempfile.TemporaryDirectory() as temp_dir:
    temp_archive_path = path.join(temp_dir, archive_name)

    # コマンドを定義
    commands = f"""
    # ターゲットフォルダをtarでアーカイブ
    tar -czf {temp_archive_path} -C {path.dirname(target)} {path.basename(target)}

    # アーカイブをリモートサーバーにアップロード
    scp {temp_archive_path} {ssh_remote}:~/

    # リモートサーバーで展開先ディレクトリを作成
    ssh {ssh_remote} "mkdir -p ~/public_html/{current_date}.stg.koemade.net"

    # リモートサーバーでアーカイブを展開（koemadeフォルダをスキップ）
    ssh {ssh_remote} "tar -xzf ~/{archive_name} --strip-components=1 -C ~/public_html/{current_date}.stg.koemade.net"

    # リモートサーバー上のアーカイブファイルを削除
    ssh {ssh_remote} "rm ~/{archive_name}"
    """

    # コマンドを実行
    for command in commands.strip().split("\n"):
        if command.strip() and not command.strip().startswith("#"):  # 空行とコメントを無視
            print(f"実行中: {command}")
            subprocess.run(command, shell=True, check=True)

print(
    f"アップロードと展開が完了しました: {ssh_remote}:~/public_html/{current_date}.stg.koemade.net")
