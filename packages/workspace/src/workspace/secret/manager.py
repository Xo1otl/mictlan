import tarfile
from typing import Protocol, List
from workspace import path, config
import os
import fnmatch


class PathRepo(Protocol):
    def listAll(self) -> List[path.Path]:
        ...


class SecretsFileRepo(PathRepo):
    def __init__(self, pattern_file: path.Path) -> None:
        self.pattern_file = pattern_file
        self.pos_patterns: List[str] = []
        self.neg_patterns: List[str] = []
        self._load_patterns()

    def _load_patterns(self) -> None:
        with open(self.pattern_file.abs(), 'r') as f:
            lines = [line.strip() for line in f if line.strip()
                     and not line.strip().startswith("#")]
        for line in lines:
            if line.startswith("!"):
                self.neg_patterns.append(line[1:])
            else:
                self.pos_patterns.append(line)

    def listAll(self) -> List[path.Path]:
        results: List[path.Path] = []
        for root, dirs, files in os.walk(config.WORKSPACE_ROOT, topdown=True):
            rel_root = os.path.relpath(root, config.WORKSPACE_ROOT)
            if rel_root == '.':
                rel_root = ''

            for d in dirs[:]:
                rel_path = os.path.join(rel_root, d) if rel_root else d
                if any(fnmatch.fnmatch(rel_path, pat) for pat in self.neg_patterns):
                    dirs.remove(d)
                    continue
                if any(fnmatch.fnmatch(rel_path, pat) for pat in self.pos_patterns):
                    full_path = os.path.join(root, d)
                    results.append(path.Path(full_path))
                    dirs.remove(d)

            for file_name in files:
                rel_path = os.path.join(
                    rel_root, file_name) if rel_root else file_name
                if any(fnmatch.fnmatch(rel_path, pat) for pat in self.neg_patterns):
                    continue
                if any(fnmatch.fnmatch(rel_path, pat) for pat in self.pos_patterns):
                    full_path = os.path.join(root, file_name)
                    results.append(path.Path(full_path))
        return results


class Packer(Protocol):
    def pack(self, files: List[path.Path], archivepath: path.Path) -> None:
        ...

    def unpack(self, archivepath: path.Path) -> None:
        ...


class TarPacker(Packer):
    def pack(self, files: List[path.Path], archivepath: path.Path) -> None:
        original_cwd = os.getcwd()
        try:
            os.chdir(config.WORKSPACE_ROOT)
            with tarfile.open(archivepath.abs(), "w") as tar:
                for p in files:
                    tar.add(str(p), recursive=True)
        finally:
            os.chdir(original_cwd)

    def unpack(self, archivepath: path.Path) -> None:
        original_cwd = os.getcwd()
        try:
            os.chdir(config.WORKSPACE_ROOT)
            with tarfile.open(archivepath.abs(), "r") as tar:
                tar.extractall()
        finally:
            os.chdir(original_cwd)


class Manager():
    def __init__(self, repo: PathRepo, packer: Packer) -> None:
        self.repo = repo
        self.packer = packer

    def import_(self, archivepath: path.Path) -> None:
        self.packer.unpack(archivepath)

    def export(self, archivepath: path.Path) -> None:
        files = self.repo.listAll()
        self.packer.pack(files, archivepath)
