import subprocess
from infra import ecosystem
from util import workspace

# Docker Composeを生成
ecosystem.generate_docker_compose()

# infraディレクトリ内の.tpl.pyファイルを取得
paths = workspace.globpaths('infra/infra/**/*.tpl.py')

# 取得したすべてのファイルをシェルで実行
for path in paths:
    subprocess.run(['python', path])  # 各ファイルをPythonシェルで実行
