import os
import uuid
import json


class DocumentFormatter:
    def _normalize_source(self, source):
        """ソースコードの内容を正規化する"""
        if isinstance(source, list):
            # リストの場合は各要素を結合して1つの文字列にする
            return ''.join(source)
        return source

    def _clean_notebook_content(self, content: str) -> str:
        """ipynbファイルから不要なメタデータを削除し、日本語を正規化する"""
        try:
            notebook = json.loads(content)

            # セルの内容だけを抽出し、不要なメタデータを削除
            cleaned_cells = []
            for cell in notebook.get('cells', []):
                # sourceの内容を正規化
                source = self._normalize_source(cell.get('source', []))

                cleaned_cell = {
                    'cell_type': cell.get('cell_type'),
                    'source': source  # リストではなく文字列として保存
                }

                # outputsは実行結果なので削除（画像データなども含まれる）
                if cell.get('cell_type') == 'code':
                    cleaned_cell['outputs'] = []

                cleaned_cells.append(cleaned_cell)

            # 最小限の情報だけを持つノートブックを作成
            cleaned_notebook = {
                'cells': cleaned_cells,
                'nbformat': notebook.get('nbformat', 4),
                'nbformat_minor': notebook.get('nbformat_minor', 0),
                'metadata': {
                    'kernelspec': notebook.get('metadata', {}).get('kernelspec', {})
                }
            }

            # ensure_ascii=Falseで日本語をそのまま出力
            return json.dumps(cleaned_notebook, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return content

    def format(self, filepath: str, content: str) -> dict:
        """検索用ドキュメントを作成する"""
        ext = os.path.splitext(filepath)[1][1:] or 'no-extension'

        # ipynbファイルの場合は内容をクリーニング
        if ext == 'ipynb':
            content = self._clean_notebook_content(content)

        return {
            'id': str(uuid.uuid4()),
            'filepath': filepath,
            'content': content,
            'ext': ext
        }
