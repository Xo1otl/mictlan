from typing import Protocol, TypeVar, Generic

T = TypeVar('T', contravariant=True)
U = TypeVar('U', covariant=True)


class Connector(Generic[T, U]):
    """
    異なる2つのリポジトリ間でデータの変換と転送を行うコネクタ。

    Transformer, SrcRepository, DestRepositoryを協調させてデータ転送を行う。
    """

    def __init__(self, transformer: 'Transformer[T, U]', src_repo: 'SrcRepository[T]', dest_repo: 'DestRepository[U]'):
        self.transformer = transformer
        self.src_repo = src_repo
        self.dest_repo = dest_repo

    def transfer(self, id: str):
        """
        src_repoからdest_repoへデータを転送する。
        """
        data: T = self.src_repo.read(id)
        converted_data: U = self.transformer.transform(data)
        self.dest_repo.save(converted_data)


class Transformer(Protocol[T, U]):
    """
    SrcRepositoryから取得したデータをDestRepositoryが保存できる形式に変換する。
    """

    def transform(self, data: T) -> U:
        ...


class SrcRepository(Protocol[U]):
    """
    変換元のデータを読み込むリポジトリ。
    データの取得を担当する。
    """

    def read(self, id: str) -> U:
        ...


class DestRepository(Protocol[T]):
    """
    変換されたデータを保存するリポジトリ。
    データの永続化を担当する。
    """

    def save(self, data: T) -> None:
        ...


class TemplateJsonFileRepository(DestRepository[dict]):
    """
    変換されたデータをJSONファイルとして保存するリポジトリ。
    データの永続化を担当する。
    """

    def save(self, data: dict) -> None:
        # JSONファイルへの保存処理を実装
        print(f"Saving data to JSON file: {data}")
