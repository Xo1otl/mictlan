#!/usr/bin/env python3

import json
import subprocess
from util import json as json_util


def fetchfs() -> dict:
    return json.loads(
        subprocess.run(
            ['tree', '-J', '.', '-L', '2', '-d'],
            capture_output=True,
            text=True
        ).stdout)[0]


def shape(fs: dict) -> dict | None:
    if fs.get('type') == 'directory' and fs.get('name') in ['__pycache__']:
        return None

    if fs.get('type') == 'file':
        return fs if fs.get('name', '').endswith('.py') else None

    if 'contents' in fs:
        new_contents = [
            result for result in (shape(child) for child in fs['contents'])
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


if __name__ == '__main__':
    fs = fetchfs()
    shaped = shape(fs)
    # print(json.dumps(shaped, cls=json_util.TreeStyleJSONEncoder))
    if shaped:
        print_tree(shaped)
