from typing import Protocol, Iterable
from typing import Tuple


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


class Processor:
    def __init__(self, collector: Collector, formatter: Formatter):
        self.collector = collector
        self.formatter = formatter

    def process(self) -> list[dict]:
        documents = []
        for filepath, content in self.collector.collect():
            document = self.formatter.format(filepath, content)
            documents.append(document)
        return documents
