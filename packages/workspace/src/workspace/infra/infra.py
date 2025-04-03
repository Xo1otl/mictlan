import inspect
import yaml
from workspace import path, config
from typing import Protocol, Dict, Callable, List, TypedDict


class Ecosystem(Protocol):
    def use(self, middleware_func: Callable[['ComposeEntry'], 'ComposeEntry']) -> None:
        ...

    def register(self, compose_func: Callable[[], Dict]) -> None:
        ...

    def configure(self) -> None:
        ...


def logging_middleware(compose_entry: 'ComposeEntry') -> 'ComposeEntry':
    print(
        f"[{compose_entry['module_name']}] Generates {compose_entry['filepath'].abs()}")
    return compose_entry


class DockerEcosystem(Ecosystem):
    def __init__(self, base_path: path.Path):
        self.base_path = base_path
        self.compose_entries: List[ComposeEntry] = []
        self.middlewares: List[Callable[[ComposeEntry], ComposeEntry]] = []

    def use(self, middleware_func: Callable[['ComposeEntry'], 'ComposeEntry']):
        self.middlewares.append(middleware_func)

    def register(self, compose_func: Callable[[], Dict]):
        func_path = path.Path(inspect.getfile(compose_func))
        compose_file_path = func_path.dir().rel2(
            self.base_path) / "docker-compose.yaml"
        content = compose_func()
        self.compose_entries.append(ComposeEntry(
            module_name=compose_func.__module__, filepath=compose_file_path, content=content))

    def configure(self):
        files_to_write = {}
        include = []

        for compose_entry in self.compose_entries:
            include.append(str(compose_entry['filepath']))
            for middleware in self.middlewares:
                compose_entry = middleware(compose_entry)
            files_to_write[compose_entry['filepath'].abs()] = yaml.dump(
                compose_entry['content'])

        base_compose = {
            'name': config.WORKSPACE_NAME,
            'include': include
        }
        base_yaml = yaml.dump(base_compose)
        files_to_write[(
            self.base_path / "docker-compose.yaml").abs()] = base_yaml

        for filepath, content in files_to_write.items():
            with open(filepath, 'w') as f:
                f.write(content)


class ComposeEntry(TypedDict):
    module_name: str
    filepath: path.Path
    content: Dict
