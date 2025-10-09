from typing import List, Any, Optional
import time


class FunSearchResult:
    """FunSearch実行結果を表すデータクラス"""

    def __init__(self, formula: str, params: str, insights: str,
                 max_nparams: int, max_mutations: int):
        self.formula = formula
        self.params = params
        self.insights = insights
        self.max_nparams = max_nparams
        self.max_mutations = max_mutations
        self.top_functions: List[tuple] = []  # (score, function_code) のリスト
        self.evaluation_count = 0
        self.mutation_count = 0
        self.start_time = time.time()
        self.end_time: Optional[float] = None

    def add_function(self, score: Any, function_code: str):
        """関数を追加してトップ10を維持"""
        self.top_functions.append((score, function_code))
        # スコアでソート（降順）してトップ10を保持
        self.top_functions.sort(key=lambda x: str(x[0]), reverse=True)
        self.top_functions = self.top_functions[:10]

    def set_counters(self, evaluation_count: int, mutation_count: int):
        """カウンターを設定"""
        self.evaluation_count = evaluation_count
        self.mutation_count = mutation_count

    def finish(self):
        """実行終了時間を設定"""
        self.end_time = time.time()
