from typing import Protocol


class Connector:
    ...


class Transformer(Protocol):
    def parse(self):
        ...


class SrcRepository(Protocol):
    def read(self):
        ...


class DestRepository(Protocol):
    def save(self):
        ...
