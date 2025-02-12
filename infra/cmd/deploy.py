#!/usr/bin/env python

import os
from workspace import config

# プロジェクトルートを見つける
root = config.WORKSPACE_ROOT

# ローカルの変数
target = 'infra'
archive_name = f'{target}.tar.gz'
archive_path = os.path.join("/tmp", archive_name)

# リモートの変数
server_alias = "ossekai"
remote_dir = "/home/ubuntu"

# 圧縮するコマンドを実行
os.system(f"tar -czf {archive_path} -C {root} {target}")
print(f"Created archive: {archive_path}")

# scp でリモートサーバーに送信
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
