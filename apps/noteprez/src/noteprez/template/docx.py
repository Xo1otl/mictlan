from .connector import *


class DocxTransformer(Transformer):
    """
    SrcRepositoryから取得したデータをDestRepositoryが保存できる形式に変換する。
    """

    def transform(self, data):
        raise NotImplementedError


class DocxFileRepository(SrcRepository):
    """
    変換元のデータを読み込むリポジトリ。
    データの取得を担当する。
    """

    def read(self, id):
        ...
