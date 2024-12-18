from . import Dataset
from typing import Protocol, Dict


class QueryRepo(Protocol):
    def categories(self) -> Dict[str, str]:
        ...

    def dataset(self, category) -> Dataset:
        ...
