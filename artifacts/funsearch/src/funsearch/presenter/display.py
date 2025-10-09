from typing import Optional, Any, Dict
from dataclasses import dataclass


@dataclass
class FunctionDisplayInfo:
    """可視化タブで表示する関数情報を管理するクラス"""

    function_name: str
    code: str
    skeleton_info: dict
    math_expression: Optional[str] = None

    def set_math_expression(self, math_expr: str) -> None:
        """数式表現を設定"""
        self.math_expression = math_expr

    def has_math_expression(self) -> bool:
        """数式表現が既に設定されているかチェック"""
        return self.math_expression is not None

    def to_markdown(self) -> str:
        """Markdown形式の表示文字列を生成"""
        md = f"## {self.function_name}\n\n```python\n{self.code}\n```"

        if self.math_expression:
            md += f"\n\n### 数式表現\n$\\large {self.math_expression}$\n"

        return md

    @classmethod
    def from_skeleton_info(cls, skeleton_info: dict) -> 'FunctionDisplayInfo':
        """skeleton_infoから FunctionDisplayInfo を作成"""
        skeleton = skeleton_info['skeleton']
        return cls(
            function_name=skeleton_info['description'],
            code=str(skeleton),
            skeleton_info=skeleton_info
        )


class SessionDisplayManager:
    """セッションごとの表示情報を管理するクラス - 全関数情報を永続保持"""

    def __init__(self):
        self.all_functions: Dict[int, FunctionDisplayInfo] = {}  # 全関数情報を永続保持
        self.selected_indices: list[int] = []  # 現在選択されている関数のindexリスト

    def add_or_update_function(self, skeleton_info: dict) -> None:
        """関数情報を追加または更新（既存の数式表現は保持）"""
        idx = skeleton_info['index']
        if idx in self.all_functions:
            # 既存の関数は数式表現を保持したまま他の情報を更新
            existing_math = self.all_functions[idx].math_expression
            self.all_functions[idx] = FunctionDisplayInfo.from_skeleton_info(
                skeleton_info)
            if existing_math:
                self.all_functions[idx].set_math_expression(existing_math)
        else:
            # 新規関数
            self.all_functions[idx] = FunctionDisplayInfo.from_skeleton_info(
                skeleton_info)

    def set_selected_functions(self, skeleton_infos: list) -> None:
        """選択された関数群を設定（全関数情報は永続保持）"""
        # まず全ての関数情報を保存/更新
        for skeleton_info in skeleton_infos:
            self.add_or_update_function(skeleton_info)

        # 選択状態を更新
        self.selected_indices = [info['index'] for info in skeleton_infos]

    def get_functions_needing_math(self) -> list[int]:
        """選択されているが数式表現がまだない関数のindexリストを取得"""
        return [idx for idx in self.selected_indices
                if idx in self.all_functions and not self.all_functions[idx].has_math_expression()]

    def add_math_expressions(self, expressions_dict: Dict[int, str]) -> None:
        """複数の関数に数式表現を一括追加（既に数式があるものはスキップ）"""
        for idx, expr in expressions_dict.items():
            if idx in self.all_functions and not self.all_functions[idx].has_math_expression():
                self.all_functions[idx].set_math_expression(expr)

    def get_current_markdown(self) -> str:
        """現在選択されている関数群の表示情報をMarkdown形式で取得"""
        if not self.selected_indices:
            return "関数を選択してください"

        markdown_parts = []
        for idx in sorted(self.selected_indices):
            if idx in self.all_functions:
                display_info = self.all_functions[idx]
                markdown_parts.append(display_info.to_markdown())

        return "\n\n---\n\n".join(markdown_parts)

    def has_selected_functions(self) -> bool:
        """現在選択されている関数があるかチェック"""
        return len(self.selected_indices) > 0

    def get_selected_indices(self) -> list[int]:
        """選択されている関数のindexリストを取得"""
        return self.selected_indices.copy()
