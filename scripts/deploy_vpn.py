import os
from util import workspace

# プロジェクトルートを見つける
root = workspace.findroot()

server_alias = "vpn"
build_dir = os.path.join(root, 'build')
archive_name = 'build.tar.gz'
archive_path = os.path.join("/tmp", 'build.tar.gz')

# リモートディレクトリの変数
remote_dir = "/home/ubuntu"

# 圧縮するコマンドを実行
os.system(f"tar -czf {archive_path} -C {root} build")
print(f"Created archive: {archive_path}")

# scp でリモートサーバーに送信
os.system(f"scp {archive_path} {server_alias}:{remote_dir}/")
print(f"Copied {archive_path} to {server_alias}:{remote_dir}/")
os.system(f"rm {archive_path}")

# リモートサーバーでの展開と処理（mictlan-buildフォルダ内に展開）
remote_commands = f"""
ssh {server_alias} << 'EOF'
    tar -xzf {remote_dir}/{archive_name} -C {remote_dir}
    echo "Files extracted to {remote_dir}."
    rm {remote_dir}/{archive_name}
EOF
"""
os.system(remote_commands)
