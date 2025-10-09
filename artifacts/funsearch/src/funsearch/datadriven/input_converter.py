from google import genai
import re
from typing import Dict, Any, Optional


class InputConverter:
    def __init__(self, client: genai.Client):
        self.models = client.models

    def _build_prompt(self, formula_text: str, theory_explanation: str, constants_description: str, variables_description: str, insights_text: str) -> str:
        prompt = f"""You are an expert AI that converts natural language descriptions of mathematical formulas into a specific structured text format usable by a code evolution tool called FunSearch.

## Input Information

### 1. Base Theoretical Formula
```
{formula_text}
```

### 2. Explanation of the Theoretical Formula
```
{theory_explanation}
```

### 3. Description of Constants (values that must not be changed during evolution)
```
{constants_description}
```

### 4. Description of Input Variables
```
{variables_description}
```

### 5. Other Insights for Evolution
```
{insights_text}
```

## Task

Carefully analyze the input information and generate a structured text output with the following sections, delimited by the specified tags. Your output must follow this structure exactly:
[DOCSTRING]
(Docstring content)
[/DOCSTRING]
[PYTHON_CODE]
(Python code content)
[/PYTHON_CODE]
[PROMPT_COMMENT]
(Prompt comment content)
[/PROMPT_COMMENT]

### Section 1: Docstring
- **Tag:** `[DOCSTRING]` and `[/DOCSTRING]`
- **Content:** Generate ONLY the docstring content for the `equation` function, formatted in Google Python Style. The docstring must remain valid after the function's logic has been evolved. **Do not mention the initial formula or its name or quote or code.** The docstring must only explain:
    - The function's overall purpose.
    - The arguments (`Args:`), including input variables and the `params` array.
    - The return value (`Returns:`).
    - The fixed constants (`Notes:`).

### Section 2: Python Code
- **Tag:** `[PYTHON_CODE]` and `[/PYTHON_CODE]`
- **Content:** Write a complete Python function named `equation`.
    - The initial function signature must accept the input variables (e.g., `x: np.ndarray`) and an additional `params: np.ndarray` for FunSearch to use.
    - The initial function body must implement the "Base Theoretical Formula".
    - The constants from "Description of Constants" should be defined and used within the function.

### Section 3: Prompt Comment
- **Tag:** `[PROMPT_COMMENT]` and `[/PROMPT_COMMENT]`
- **Content:** Create an instruction comment for FunSearch's LLM.
    - It should state the original mathematical function.
    - It should provide context from the formula explanation, variable descriptions, and other insights.

Now, start generating the structured text output.
"""
        return prompt

    def _send_request(self, prompt: str) -> str:
        try:
            response = self.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            if hasattr(response, 'text') and response.text:
                return response.text
            else:
                raise ValueError(
                    "Gemini returned no text or an unexpected response structure.")

        except Exception as e:
            error_details = ""
            if hasattr(e, 'response') and hasattr(e.response, 'text'):  # type: ignore
                error_details = e.response.text  # type: ignore
            elif hasattr(e, 'message'):
                error_details = e.message  # type: ignore
            print(
                f"Error during Gemini API call: {e}\nDetails: {error_details}")
            raise

    def _parse_response(self, response_text: str) -> Dict[str, str]:
        try:
            print(f"Attempting to parse response: {response_text[:500]}...")

            docstring_match = re.search(
                r"\[DOCSTRING\](.*?)\[/DOCSTRING\]", response_text, re.DOTALL)
            python_code_match = re.search(
                r"\[PYTHON_CODE\](.*?)\[/PYTHON_CODE\]", response_text, re.DOTALL)
            prompt_comment_match = re.search(
                r"\[PROMPT_COMMENT\](.*?)\[/PROMPT_COMMENT\]", response_text, re.DOTALL)

            if not (docstring_match and python_code_match and prompt_comment_match):
                raise ValueError(
                    "One or more sections were not found in the LLM response.")

            docstring = docstring_match.group(1).strip()
            python_code = python_code_match.group(1).strip()
            prompt_comment = prompt_comment_match.group(1).strip()

            # --- Start Cleaning Process ---

            # General cleaner for markdown code blocks
            def clean_markdown_code_block(text, language='python'):
                text = text.strip()
                if text.startswith(f'```{language}'):
                    text = text[len(f'```{language}'):].strip()
                elif text.startswith('```'):
                    text = text[3:].strip()
                if text.endswith('```'):
                    text = text[:-3].strip()
                return text

            # Clean docstring
            docstring = clean_markdown_code_block(docstring)
            if docstring.startswith('"""') and docstring.endswith('"""'):
                docstring = docstring[3:-3].strip()

            # Clean python_code and extract from 'def'
            python_code = clean_markdown_code_block(python_code)
            def_pos = python_code.find('def ')
            if def_pos != -1:
                python_code = python_code[def_pos:]

            # Clean prompt_comment
            prompt_comment = clean_markdown_code_block(
                prompt_comment, language='')  # language agnostic
            lines = prompt_comment.splitlines()
            cleaned_lines = [line.lstrip().lstrip('#').strip()
                             for line in lines]
            prompt_comment = '\n'.join(cleaned_lines)

            # --- End Cleaning Process ---

            return {
                "docstring": docstring,
                "equation_src": python_code,
                "prompt_comment": prompt_comment,
            }

        except (AttributeError, ValueError) as e:
            raise ValueError(
                f"Failed to parse LLM response: {e}. Response text: {response_text}")

    def convert(self, formula_text: str, theory_explanation: str, constants_description: str, variables_description: str, insights_text: str) -> Optional[Dict[str, Any]]:
        """
        変換プロセス全体を実行します。
        """
        try:
            print("--- Starting conversion process (Text Parsing Approach) ---")
            print(f"Formula Text (snippet): {formula_text[:100]}...")
            print(
                f"Variables Text (snippet): {variables_description[:100]}...")
            print(f"Insights Text (snippet): {insights_text[:100]}...")

            prompt = self._build_prompt(
                formula_text, theory_explanation, constants_description, variables_description, insights_text)
            # print(f"Generated Prompt:\n{prompt}") # Too long, disable for now

            raw_response = self._send_request(prompt)
            print(f"Received Raw Response:\n{raw_response}")

            final_results = self._parse_response(raw_response)

            print("--- Conversion successful (Text Parsing Approach) ---")
            return final_results
        except Exception as e:
            print(f"An error occurred during the conversion process: {e}")
            return None
