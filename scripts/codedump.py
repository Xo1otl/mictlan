#!/usr/bin/env python3

import argparse
from util import localfile


def main():
    parser = argparse.ArgumentParser(
        description='指定されたフォルダ内の特定の拡張子を持つファイルの内容を表示する')
    parser.add_argument('folder', type=str, help='対象フォルダのパス')
    parser.add_argument('extension', type=str, default=".py",
                        help='対象ファイルの拡張子 (例: .py, .txt), デフォルトは.py')
    args = parser.parse_args()

    localfile.print_source(args.folder, args.extension)


if __name__ == "__main__":
    main()
