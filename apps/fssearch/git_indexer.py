from infra import searchengine
import json
import uuid
import shutil
import tempfile
import fnmatch
from typing import Optional, Iterable, Tuple, List
import git
import os
import meilisearch
from typing import Protocol, Iterable
from typing import Tuple


# Interfaces
class Collector(Protocol):
    def collect(self) -> Iterable[Tuple[str, str]]:
        """
        コンテンツの収集を行う
        Returns:
            Iterable[Tuple[str, str]]: (filepath, content)のタプルのイテラブル
        """
        ...


class Formatter(Protocol):
    def format(self, filepath: str, content: str) -> dict:
        """
        コンテンツを検索用データに変換する
        Args:
            filepath: ファイルパス
            content: ファイルの内容
        Returns:
            dict: 検索用のドキュメント
        """
        ...


class Uploader(Protocol):
    def upload(self, documents: list[dict]) -> None:
        """
        検索エンジンにデータを登録する
        Args:
            documents: アップロード対象のドキュメントリスト
        """
        ...


# Domain Classes
class FileProcessor:
    def __init__(self, collector: Collector, formatter: Formatter):
        self.collector = collector
        self.formatter = formatter

    def process(self) -> list[dict]:
        documents = []
        for filepath, content in self.collector.collect():
            document = self.formatter.format(filepath, content)
            documents.append(document)
        return documents


class Indexer:
    def __init__(self, processor: FileProcessor, uploader: Uploader):
        self.processor = processor
        self.uploader = uploader

    def index(self) -> None:
        documents = self.processor.process()
        if documents:
            self.uploader.upload(documents)


def create_indexer(collector: Collector, formatter: Formatter, uploader: Uploader) -> Indexer:
    processor = FileProcessor(collector, formatter)
    return Indexer(processor, uploader)


class MeilisearchUploader:
    def __init__(self, host: str, api_key: str, index_name: str):
        self.client = meilisearch.Client(host, api_key)
        self.index = self.client.index(index_name)

        # 検索可能なフィールドとフィルタリング属性を設定
        self.index.update_settings({
            'searchableAttributes': [
                'content',
                'filepath'
            ],
            'filterableAttributes': [
                'ext'
            ]
        })

    def upload(self, documents: list[dict]) -> None:
        """ドキュメントをMeilisearchにアップロードする"""
        if documents:
            self.index.add_documents(documents)
            print(f"Indexed {len(documents)} files successfully")
        else:
            print("No text files found to index")


class GitCollector:
    def __init__(self, repo_path: str, ignore_patterns: List[str] = []):
        """
        Args:
            repo_path: GitリポジトリのパスまたはURL
                      - ローカル: '/path/to/repo' や 'C:\\path\\to\\repo'
                      - リモート: 'https://github.com/user/repo.git' や 'git@github.com:user/repo.git'
            ignore_patterns: 除外するファイルのglobパターンのリスト
                           例: ['*.pdf', 'test/**/*.py', 'tmp/*']
        """
        self.is_remote = repo_path.startswith(
            ('http://', 'https://', 'git@', 'ssh://'))

        if self.is_remote:
            self.temp_dir = tempfile.mkdtemp()
            self.repo = git.Repo.clone_from(repo_path, self.temp_dir)
            self.repo_path = self.temp_dir
        else:
            self.repo_path = repo_path
            self.repo = git.Repo(repo_path)
            self.temp_dir = None

        self.ignore_patterns = ignore_patterns or []

    def _should_ignore(self, file_path) -> bool:
        """
        ファイルを無視すべきかどうかを判定する

        Args:
            file_path: 判定対象のファイルパス（リポジトリルートからの相対パス）

        Returns:
            bool: 無視すべき場合はTrue
        """
        # Windowsのパス区切り文字をUNIX形式に統一
        normalized_path = file_path.replace(os.sep, '/')

        for pattern in self.ignore_patterns:
            # パターンもUNIX形式に統一
            normalized_pattern = pattern.replace(os.sep, '/')

            # **/ で始まるパターンの場合は、すべてのサブディレクトリにマッチ
            if pattern.startswith('**/'):
                if fnmatch.fnmatch(normalized_path, pattern[3:]):
                    return True

            # パターンに / が含まれる場合は、完全パスでマッチング
            if '/' in normalized_pattern:
                if fnmatch.fnmatch(normalized_path, normalized_pattern):
                    return True
            else:
                # パターンに / が含まれない場合は、ファイル名のみでマッチング
                if fnmatch.fnmatch(os.path.basename(normalized_path), normalized_pattern):
                    return True

        return False

    def _read_file_content(self, file_path) -> Optional[str]:
        """ファイルの内容を読み込む。バイナリファイルの場合はNoneを返す"""
        try:
            with open(os.path.join(self.repo_path, file_path), 'r', encoding='utf-8') as f:
                return f.read()
        except (UnicodeDecodeError, IOError):
            return None

    def collect(self) -> Iterable[Tuple]:
        """Git管理下のファイルとその内容を収集する（ignore_patternsに一致するファイルは除外）"""
        tracked_files = [item[0] for item in self.repo.index.entries]

        for file_path in tracked_files:
            if not self._should_ignore(file_path):
                content = self._read_file_content(file_path)
                if content is not None:
                    yield file_path, content

    def __del__(self):
        """デストラクタ: リモートの場合、一時ディレクトリを削除"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


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


collector = GitCollector('/workspaces/mictlan', ignore_patterns=[
    "*.min.css",
    "*-min.css",
    "*.sum",
    "known_hosts",
    "cdk.json",
    "koemadeinfo.html"
])
formatter = DocumentFormatter()
uploader = MeilisearchUploader(
    'http://meilisearch:7700', searchengine.MEILI_MASTER_KEY, 'mictlan')
create_indexer(collector, formatter, uploader).index()
