from .domain import *
import ast
import jax.numpy as jnp
import numpy as np
import scipy


# def の上に """AAA""" みたいなコメントつけるのはナシ parse に失敗するし inspect.getsource でも取れない
# 処理は LLM-SR の実装に従う
# 1. 関数部分だけを取り出した text をもとに構成する
# 2. ast.parse で関数をパースする
# 3. 元のコードではめちゃくちゃ頑張って docstring パースして戻すなどしていたけど多分無駄
# TODO: こっちで関数名を変更できると嬉しい
class PyAstSkeleton(Skeleton):
    def __init__(self, fn_code: str):
        try:
            node = ast.parse(fn_code)
            code_obj = compile(node, filename="<ast>", mode="exec")
        except SyntaxError as e:
            raise ValueError("提供されたソースコードのパースに失敗しました", fn_code) from e

        # TODO: scipy の細かい関数なども名前空間に追加しといたほうがいいかも
        local_ns = {}
        local_ns['np'] = np # FIXME: 一時的に numpy に戻している
        local_ns['scipy'] = scipy  # scipy を追加
        exec(code_obj, local_ns)

        # 関数定義（FunctionDef）であることを確認
        if not node.body or not isinstance(node.body[0], ast.FunctionDef):
            raise ValueError("提供されたソースコードに関数定義が見つかりません", fn_code)
        func_name = node.body[0].name

        # コンパイル済みの名前空間から関数オブジェクトを取得し、引数をそのまま渡して実行します
        self._func = local_ns[func_name]
        self._fn_code = fn_code

    def __call__(self, *args: Any, **kwargs: Any):
        try:
            result = self._func(*args, **kwargs)
            return result
        except Exception as e:
            raise RuntimeError("関数の実行中にエラーが発生しました", self._fn_code) from e

    def __str__(self):
        return self._fn_code
