from typing import Protocol


class Engine:
    def __init__(self, template_repo: 'TemplateRepository', mapper: 'Mapper'):
        ...


class TemplateRepository(Protocol):
    ...


class Mapper(Protocol):
    ...
