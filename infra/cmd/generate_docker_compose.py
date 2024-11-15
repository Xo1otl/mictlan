#!/usr/bin/env python

import subprocess
from infra import ecosystem
from util import workspace

# infraディレクトリ内の.tpl.pyファイルを取得
paths = workspace.globpaths('infra/infra/**/*.tpl.py')

# 取得したすべてのファイルをシェルで実行
for path in paths:
    subprocess.run(['python', path])  # 各ファイルをPythonシェルで実行

# Docker Composeを生成(tplを実行した後に生成する必要がある)
ecosystem.gen_compose()
