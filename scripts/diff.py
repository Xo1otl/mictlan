#!/usr/bin/env python3

import subprocess
import difflib
import sys


# コマンドを実行して結果を取得する関数
def run_command(command):
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        if result.returncode != 0:
            print(f"Error running command '{command}': {result.stderr}")
            return None
        return result.stdout
    except Exception as e:
        print(f"Exception occurred while running command: {e}")
        return None


# メインの処理
def main():
    if len(sys.argv) != 3:
        print("Usage: ./script.py '<command1>' '<command2>'")
        sys.exit(1)

    # 引数から2つのコマンドを取得
    command1 = sys.argv[1]
    command2 = sys.argv[2]

    # それぞれのコマンドを実行
    output1 = run_command(command1)
    output2 = run_command(command2)

    # 結果が取得できているか確認
    if output1 is not None and output2 is not None:
        # 差分を取得
        diff = difflib.unified_diff(
            output1.splitlines(), output2.splitlines(), lineterm='')

        # 差分を表示
        print('\n'.join(diff))
    else:
        print("Error: Could not retrieve output for one or both commands.")


if __name__ == "__main__":
    main()
