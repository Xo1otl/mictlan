#!/usr/bin/env python

import os
from util import workspace

# プロジェクトルートを見つける
root = workspace.findroot()

# ローカルの変数
targets = ['apps/zaiko/', 'infra']  # 配列でファイルやフォルダを指定
archive_name = 'archive.tar.gz'
archive_path = os.path.join("/tmp", archive_name)

# リモートの変数
server_alias = "awsjob"
remote_dir = "/home/ec2-user/mictlan"

# 圧縮するコマンドを構築
target_paths = f" -C {root} {' '.join(targets)}"  # ターゲットを結合してコマンドに渡す
os.system(f"tar -czf {archive_path} {target_paths}")
print(f"Created archive: {archive_path}")

# リモートサーバーで mictlan フォルダを作成し、scp でリモートサーバーに送信
os.system(f"ssh {server_alias} 'mkdir -p {remote_dir}'")
os.system(f"scp {archive_path} {server_alias}:{remote_dir}/")
print(f"Copied {archive_path} to {server_alias}:{remote_dir}/")
os.system(f"rm {archive_path}")

# リモートサーバーでの展開と処理
remote_commands = f"""
ssh {server_alias} << 'EOF'
    tar -xzf {remote_dir}/{archive_name} -C {remote_dir}
    echo "Files extracted to {remote_dir}."
    rm {remote_dir}/{archive_name}
EOF
"""
os.system(remote_commands)
