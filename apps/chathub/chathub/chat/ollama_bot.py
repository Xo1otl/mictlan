from typing import Dict, List, Any, Generator
from ollama import Client, Message


class OllamaBot:
    def __init__(self, host: str = "http://ollama:11434"):
        self.client = Client(host=host)

    def chat(self, messages: List[Dict[str, str]]) -> Generator[Dict[str, Any], None, None]:
        ollama_messages = [
            Message(role=m['role'], content=m['content']) for m in messages]  # type: ignore
        stream = self.client.chat(
            model='jaahas/gemma-2-9b-it-abliterated',  # モデル名は適宜変更
            messages=ollama_messages,
            stream=True
        )

        for chunk in stream:
            if chunk.get('message', {}).get('content'):
                yield chunk  # type: ignore
