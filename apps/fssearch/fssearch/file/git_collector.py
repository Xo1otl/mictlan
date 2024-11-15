import os
import git
from typing import Optional, Iterable, Tuple, List
import tempfile
import shutil
from fnmatch import fnmatch


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
        return any(fnmatch(file_path, pattern) for pattern in self.ignore_patterns)

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
