from funsearch.presenter.math_expression_generator import MathExpressionGenerator
import os
from google import genai


class DummySkeleton:
    def __init__(self, code: str):
        self.code = code
    
    def __str__(self) -> str:
        return self.code


def main():
    try:
        api_key = os.environ["GOOGLE_CLOUD_API_KEY"]
    except KeyError:
        from infra.ai import llm
        api_key = llm.GOOGLE_CLOUD_API_KEY
    
    gemini_client = genai.Client(api_key=api_key)
    generator = MathExpressionGenerator(gemini_client)
    
    # テスト用のスケルトンを作成
    test_skeletons = [
        DummySkeleton("def evaluate(inputs, params):\n    return inputs[0] * params[0] + params[1]"),
        DummySkeleton("def evaluate(inputs, params):\n    return inputs[0] ** 2 + params[0]"),
        DummySkeleton("def evaluate(inputs, params):\n    return params[0] * inputs[0] * inputs[1] + params[1]"),
    ]
    
    print("Testing MathExpressionGenerator...")
    print(f"Number of test skeletons: {len(test_skeletons)}")
    print()
    
    for i, skeleton in enumerate(test_skeletons):
        print(f"Skeleton {i+1}:")
        print(skeleton)
        print()
    
    try:
        expressions = generator.generate_expressions(test_skeletons)
        print("Generated LaTeX expressions:")
        for i, expr in enumerate(expressions):
            print(f"{i+1}: {expr}")
    except Exception as e:
        print(f"Error generating expressions: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
