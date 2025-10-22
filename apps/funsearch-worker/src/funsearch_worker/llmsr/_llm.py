from funsearch_worker import propose


class LLM(propose.LLM):
    def generate(self, prompt: str) -> str:
        return prompt
