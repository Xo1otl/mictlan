import os
from typing import List, Optional


def print_source(directory: str, ext: str) -> Optional[List[str]]:
    printed_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                file_path = os.path.join(root, file)
                printed_files.append(file_path)

                print(f"\n# {file_path}")
                print(f"```")

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(content)
                        print("```")
                except Exception as e:
                    print(f"エラー: ファイル '{file_path}' を読み込めませんでした。理由: {str(e)}")

    return printed_files if printed_files else None
