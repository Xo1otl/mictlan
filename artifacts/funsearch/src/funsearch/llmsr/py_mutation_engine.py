from funsearch import function
from typing import List, Callable
from google import genai
from pydantic import BaseModel
import requests
import json
import textwrap
from .code_manipulation import *


class ResponseSchema(BaseModel):
    improved_equation: str


class PyMutationEngine(function.MutationEngine):
    def __init__(self, prompt_comment: str, docstring: str, gemini_client: None | genai.Client = None, max_nparams: int = 10):
        self._profilers: List[Callable[[
            function.MutationEngineEvent], None]] = []
        self._prompt_comment = prompt_comment
        self._docstring = docstring
        self._gemini_client = gemini_client
        self._max_nparams = max_nparams

    def _ask_gemini(self, prompt: str) -> str:
        if self._gemini_client is None:
            raise Exception("Gemini client is not initialized.")
        response = self._gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': ResponseSchema,
            },
        )
        try:
            parsed: ResponseSchema = response.parsed  # type: ignore
            improved_equation = parsed.improved_equation
            return improved_equation
        except Exception as e:
            raise Exception(
                f"gemini response parse error",
                response.text
            ) from e

    def _ask_ollama(self, prompt: str) -> str:
        url = "http://ollama:11434/api/generate"
        payload = {
            "prompt": prompt,
            # "model": "gemma3:12b", # メモリリークする
            # "model": "phi4",
            "model": "qwen2.5-coder:14b",
            # "model": "hf.co/google/gemma-3-12b-it-qat-q4_0-gguf", # やっぱりメモリリークする
            "format": ResponseSchema.model_json_schema(),
            "stream": False,
            # モデルによってはエラー爆増する
            # "options": {
            #     "temperature": 1,  # ちょっと高めがいい気がする
            # }
        }
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        generated_text = result["response"]
        parsed_output = json.loads(generated_text)
        improved_equation = parsed_output["improved_equation"]
        return improved_equation

    def mutate(self, fn_list: List[function.Function]):
        for profiler_fn in self._profilers:
            profiler_fn(function.OnMutate(type="on_mutate", payload=fn_list))
        # スコアの順に並べる
        sorted_fn_list = sorted(
            fn_list,
            key=lambda fn: fn.score()
        )
        skeletons = [fn.skeleton() for fn in sorted_fn_list]
        prompt = self._construct_prompt(skeletons)
        # これは時間がかかる処理
        answer = self._ask_gemini(prompt)
        fn_code = self._parse_answer(answer, str(skeletons[0]))
        new_skeleton = function.PyAstSkeleton(fn_code)
        new_fn = fn_list[0].clone(new_skeleton)  # どれcloneしても構わん
        for profiler_fn in self._profilers:
            profiler_fn(function.OnMutated(
                type="on_mutated",
                payload=(fn_list, new_fn)
            ))
        return new_fn

    def _construct_prompt(self, skeletons: List[function.Skeleton]) -> str:
        prompt = f'''
You are a helpful assistant exploring scientific mathematical functions. Complete the Python function by **changing one or more structures** from previous versions to discover a more physically accurate solution.

"""{remove_empty_lines(self._prompt_comment)}"""

import numpy as np
import scipy

# Initialize parameters
MAX_NPARAMS = {self._max_nparams}
PRAMS_INIT = [1.0] * MAX_NPARAMS


{''.join(f"{remove_empty_lines(set_fn_name(remove_docstring(str(skeleton)), i))}\n" for i, skeleton in enumerate(skeletons))}
# Improved version of `equation_v{len(skeletons)-1}`.
def equation_v{len(skeletons)}({extract_fn_header(str(skeletons[0]))}) -> np.ndarray:
    """ 
{textwrap.indent(self._docstring.strip(), '    ')}
    """
 

Implement `equation_v{len(skeletons)}` by **modifying its calculation logic** for improvement.
'''

        return prompt

    def _parse_answer(self, answer: str, example: str) -> str:
        answer = extract_last_function(answer)  # 失敗したらそのまま後続に渡される
        answer = answer.replace('```', '')
        answer = fix_single_quote_line(answer)
        answer = fix_missing_header_and_ret(answer, example)
        answer = fix_indentation(answer)
        answer = fix_wrong_escape(answer)
        answer = force_squash_assignment_statement(answer)
        answer = force_squash_return_statement(answer)
        return answer

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)


class PyMutationEngineUnstructured(function.MutationEngine):
    def __init__(self, prompt_comment: str, docstring: str, gemini_client: None | genai.Client = None, max_nparams: int = 10):
        self._profilers: List[Callable[[
            function.MutationEngineEvent], None]] = []
        self._prompt_comment = prompt_comment
        self._docstring = docstring
        self._gemini_client = gemini_client
        self._max_nparams = max_nparams

    def _ask_gemini(self, prompt: str) -> str:
        if self._gemini_client is None:
            raise Exception("Gemini client is not initialized.")

        response = self._gemini_client.models.generate_content(
            contents=prompt,
            model='gemini-2.0-flash',
        )

        if response.text is None:
            raise Exception("Gemini response is empty or invalid.")

        return response.text

    def mutate(self, fn_list: List[function.Function]):
        for profiler_fn in self._profilers:
            profiler_fn(function.OnMutate(type="on_mutate", payload=fn_list))
        # スコアの順に並べる
        sorted_fn_list = sorted(
            fn_list,
            key=lambda fn: fn.score()
        )
        skeletons = [fn.skeleton() for fn in sorted_fn_list]
        prompt = self._construct_prompt(skeletons)
        # これは時間がかかる処理
        answer = self._ask_gemini(prompt)
        fn_code = self._parse_answer(answer, str(skeletons[0]))
        new_skeleton = function.PyAstSkeleton(fn_code)
        new_fn = fn_list[0].clone(new_skeleton)  # どれcloneしても構わん
        for profiler_fn in self._profilers:
            profiler_fn(function.OnMutated(
                type="on_mutated",
                payload=(fn_list, new_fn)
            ))
        return new_fn

    def _construct_prompt(self, skeletons: List[function.Skeleton]) -> str:
        prompt = f'''
You are a helpful assistant exploring scientific mathematical functions. Complete the Python function by **changing one or more structures** from previous versions to discover a more physically accurate solution.

"""{remove_empty_lines(self._prompt_comment)}"""

import numpy as np
import scipy

# Initialize parameters
MAX_NPARAMS = {self._max_nparams}
PRAMS_INIT = [1.0] * MAX_NPARAMS


{''.join(f"{remove_empty_lines(set_fn_name(remove_docstring(str(skeleton)), i))}\n" for i, skeleton in enumerate(skeletons))}
# Improved version of `equation_v{len(skeletons)-1}`.
def equation_v{len(skeletons)}({extract_fn_header(str(skeletons[0]))}) -> np.ndarray:
    """ 
{textwrap.indent(self._docstring.strip(), '    ')}
    """
 

Implement `equation_v{len(skeletons)}` by **modifying its calculation logic** for improvement.
Please provide ONLY the Python code for the improved function, including the `def` header, enclosed in a code block.
'''

        return prompt

    def _parse_answer(self, answer: str, example: str) -> str:
        parsed_answer = parse_my_text(answer)
        if parsed_answer is None:
            raise Exception("Failed to parse the answer.", answer)
        return parsed_answer

    def use_profiler(self, profiler_fn):
        self._profilers.append(profiler_fn)
        return lambda: self._profilers.remove(profiler_fn)
