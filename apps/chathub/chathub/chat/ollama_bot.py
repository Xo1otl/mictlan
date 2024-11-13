from typing import Dict, List, Any, Generator, Literal, cast
from ollama import Client, Message


class OllamaBot:
    def __init__(self, host: str = "http://ollama:11434"):
        self.client = Client(host=host)

    def chat(self, messages: List[Dict[str, str]]) -> Generator[Dict[str, Any], None, None]:
        ollama_messages = [Message(
            role=cast(
                Literal['user', 'assistant', 'system', 'tool'], m['role']
            ),
            content=m['content']
        ) for m in messages]
        stream = self.client.chat(
            model='jaahas/gemma-2-9b-it-abliterated',
            messages=ollama_messages,
            stream=True
        )

        for chunk in stream:
            if chunk.get('message', {}).get('content'):
                yield dict(chunk)
