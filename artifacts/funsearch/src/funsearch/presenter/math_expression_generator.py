from typing import List
from pydantic import BaseModel
import json
from funsearch import function


class MathExpressionResponse(BaseModel):
    """Gemini APIからのstructured outputスキーマ"""
    markdown_latex_expressions: List[str]


class MathExpressionGenerator:
    """複数の関数に対してLaTeX数式表現を一括生成するクラス"""

    def __init__(self, gemini_client):
        self.client = gemini_client

    def generate_expressions(self, skeletons: List[function.Skeleton]) -> List[str]:
        """
        複数の関数に対して数式表現を一括生成

        Args:
            skeletons: 関数クラスのリスト、__str__がpythonコード表現文字列を返すのでそれをプロンプトに含めるべし

        Returns:
            [latex_expression]のリスト
        """
        if not skeletons:
            return []
            
        prompt = self._build_prompt(skeletons)
        response = self._send_request(prompt)
        return self._parse_response(response, len(skeletons))

    def _send_request(self, prompt: str) -> str:
        """Gemini APIにリクエストを送信"""
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': MathExpressionResponse,
                },
            )

            if hasattr(response, 'parsed') and response.parsed:
                return response.parsed
            elif hasattr(response, 'text') and response.text:
                # フォールバック: JSONテキストを手動でパース
                return json.loads(response.text)
            else:
                raise ValueError(
                    "Gemini returned no response or unexpected structure")

        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            raise

    def _parse_response(self, response, expected_count: int) -> List[str]:
        """レスポンスをパースしてリスト形式で返す"""
        if isinstance(response, MathExpressionResponse):
            expressions = response.markdown_latex_expressions
        elif isinstance(response, dict) and 'markdown_latex_expressions' in response:
            expressions = response['markdown_latex_expressions']
        else:
            raise ValueError(f"Unexpected response format: {type(response)}")
            
        # 期待する数だけ返す、不足している場合はフォールバックを追加
        result = expressions[:expected_count]
        while len(result) < expected_count:
            result.append(f"f(x) = [\\text{{expression_{len(result)}}}]")
            
        return result

    def _build_prompt(self, skeletons: List[function.Skeleton]) -> str:
        """数式表現生成用のプロンプトを構築"""
        function_codes = []
        for i, skeleton in enumerate(skeletons):
            function_codes.append(f"Function {i+1}:\n```python\n{str(skeleton)}\n```")
        
        functions_text = "\n\n".join(function_codes)

        return f"""Convert these Python functions to LaTeX mathematical expressions.

{functions_text}

Return exactly {len(skeletons)} LaTeX expressions in the same order."""
