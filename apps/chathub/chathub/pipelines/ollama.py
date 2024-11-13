from typing import List, Union, Generator, Iterator, Optional
import requests
import json


class Pipeline:
    def __init__(self):
        self.name = "Ollama Pipeline"
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

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(messages)
        print(f"pipe:{__name__}")

        OLLAMA_BASE_URL = "http://ollama:11434"
        MODEL = "jaahas/gemma-2-9b-it-abliterated"

        print(json.dumps(body, indent=2, ensure_ascii=False))

        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")

        try:
            r = requests.post(
                url=f"{OLLAMA_BASE_URL}/v1/chat/completions",
                json={**body, "model": MODEL},
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                full_response = []

                def response_generator():
                    for line in r.iter_lines():
                        if line:
                            try:
                                json_response = json.loads(line)
                                full_response.append(json_response)
                                yield line
                            except json.JSONDecodeError:
                                yield line
                    # イテレーションが終わった後に全体の応答を表示
                    print("\n=== Complete Response ===")
                    print(json.dumps(full_response, indent=2, ensure_ascii=False))
                    print("=======================")

                return response_generator()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"
