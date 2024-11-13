from typing import List, Union, Generator, Iterator, Optional
from chathub import chat
import json
from datetime import datetime


class Pipeline:
    def __init__(self):
        self.name = "ChatHub Pipeline"
        bot = chat.OllamaBot()
        repository = chat.GitRepository()
        self.manager = chat.Manager(bot, repository)
        pass

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        body["pipeline_metadata"] = {"chat_id": body["chat_id"]}
        return body

    def convert_stream(self, stream) -> Generator[bytes, None, None]:
        for item in stream:
            created = int(datetime.fromisoformat(
                item['created_at'].replace('Z', '+00:00')).timestamp())

            response = {
                "id": "chatcmpl-421",
                "object": "chat.completion.chunk",
                "created": created,
                "model": item["model"],
                "system_fingerprint": "fp_ollama",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": item["message"]["role"],
                        "content": item["message"]["content"]
                    },
                    "finish_reason": "stop" if item["done"] else None
                }]
            }

            yield f"data: {json.dumps(response)}".encode()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")

        try:
            stream = self.manager.mediate(
                body["user"]["id"], body["pipeline_metadata"]["chat_id"], messages)

            return self.convert_stream(stream)
        except Exception as e:
            return f"Error: {e}"
