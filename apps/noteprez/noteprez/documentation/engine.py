from typing import Protocol


class Engine:
    """
    テンプレートとノートを元にドキュメントを生成するエンジン。

    Mapper, Producer, TemplateRepository, NoteRepositoryを協調させてドキュメント生成を行う。
    """

    def __init__(self, mapper: 'Mapper', producer: 'Producer', template_repository: 'TemplateRepository', note_repository: 'NoteRepository'):
        self.mapper = mapper
        self.producer = producer
        self.template_repository = template_repository
        self.note_repository = note_repository

    def generate(self, template_id: str, note_id: str):
        """
        テンプレートとノートを元にドキュメントを生成する。

        Args:
            template_id (str): テンプレートID
            note_id (str): ノートID
        """
        template = self.template_repository.find_by_id(template_id)
        note = self.note_repository.find_by_id(note_id)
        content = self.mapper.map(template, note)
        self.producer.produce(content)


class Mapper(Protocol):
    """
    ノートから構造化されたデータを作成する役割を持つ。
    """

    def map(self, template, note):
        ...


class Producer(Protocol):
    """
    構造化されたデータからドキュメントを生成し、出力する役割を持つ。
    """

    def produce(self, content):
        ...


class TemplateRepository(Protocol):
    """
    テンプレートを読み込むリポジトリ。
    テンプレートの取得と永続化を担当する。
    """

    def find_by_id(self, template_id):
        ...


class NoteRepository(Protocol):
    """
   ノートを読み込み/書き込みするリポジトリ。
   ノートの取得と永続化を担当する。
    """

    def find_by_id(self, note_id):
        ...
