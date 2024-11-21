from typing import Protocol


class Manager:
    def __init__(self, repository: 'Repository', extractor: 'Extractor'):
        self.repository = repository
        self.extractor = extractor


class Repository(Protocol):
    def read(self):
        ...


class Extractor(Protocol):
    def extract(self):
        ...
