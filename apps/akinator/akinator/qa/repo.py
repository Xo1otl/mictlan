from . import Dataset
from typing import Protocol


class Repo(Protocol):
    def dataset(self, category) -> Dataset:
        ...
