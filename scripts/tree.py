#!/usr/bin/env python3

import json
import subprocess
import argparse


def fetchfs(directory: str, depth: int) -> dict:
    cmd = ['tree', '-J', directory, '-L', str(depth), '-d', '--gitignore']
    return json.loads(
        subprocess.run(
            cmd,
            capture_output=True,
            text=True
        ).stdout)[0]


def shape(fs: dict, extensions: list[str]) -> dict | None:
    if fs.get('type') == 'directory' and fs.get('name') in ['__pycache__']:
        return None

    if fs.get('type') == 'file':
        return fs if fs.get('name', '').endswith(tuple(extensions)) else None

    if 'contents' in fs:
        new_contents = [
            result for result in (shape(child, extensions) for child in fs['contents'])
            if result is not None
        ]
        fs['contents'] = new_contents

    return fs


def print_tree(fs: dict, indent: int = 2) -> None:
    print(" " * indent + fs["name"])
    if "contents" in fs:
        for item in fs["contents"]:
            if item.get("type") == "directory":
                item["name"] += "/"
            print_tree(item, indent + 2)


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='ファイルシステムのツリー構造を表示する')
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='ファイルシステム情報を取得するディレクトリ (デフォルト: .)'
    )
    parser.add_argument(
        '-d',
        '--depth',
        type=int,
        default=2,
        help='取得するディレクトリの深さ (デフォルト: 2)'
    )
    parser.add_argument(
        '-e',
        '--extension',
        nargs='*',
        default=['py'],
        help='対象とするファイルの拡張子 (デフォルト: py)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parseargs()
    fs = fetchfs(args.directory, args.depth)
    shaped = shape(fs, args.extension)
    if shaped:
        print_tree(shaped)
