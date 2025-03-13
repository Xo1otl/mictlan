from typing import Protocol


class Dataset(Protocol):
    def generate(self) -> None:
        ...

    def save(self, path: str) -> None:
        ...

    def preprocess(self) -> None:
        ...

    def load(self, path: str) -> None:
        ...
