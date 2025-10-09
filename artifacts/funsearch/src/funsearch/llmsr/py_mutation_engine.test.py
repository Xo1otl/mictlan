from google import genai
from funsearch import llmsr
import os


def test_py_mutation_engine():
    try:
        api_key = os.environ["GOOGLE_CLOUD_API_KEY"]
    except KeyError:
        from infra.ai import llm
        api_key = llm.GOOGLE_CLOUD_API_KEY
    
    gemini_client = genai.Client(api_key=api_key)
    engine = llmsr.PyMutationEngine(
        prompt_comment="",
        docstring="",
        gemini_client=gemini_client
    )
    response = engine._ask_gemini("1 + 1")
    # response = engine._ask_ollama("1 + 1")
    print(response)


if __name__ == '__main__':
    test_py_mutation_engine()
