from typing import Protocol
from fssearch import file


class Uploader(Protocol):
    def upload(self, documents: list[dict]) -> None:
        """
        検索エンジンにデータを登録する
        Args:
            documents: アップロード対象のドキュメントリスト
        """
        ...


class Indexer:
    def __init__(self, processor: file.Processor, uploader: Uploader):
        self.processor = processor
        self.uploader = uploader

    def index(self) -> None:
        documents = self.processor.process()
        if documents:
            self.uploader.upload(documents)
