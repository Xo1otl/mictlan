from typing import Protocol, List, Iterable, Tuple, Dict
from dataclasses import dataclass
import pandas as pd
import math


class EBT(Protocol):
    ...


class Node(Protocol):
    def groupby(self, by: str) -> Iterable[Tuple[str, Iterable, Iterable]]:
        """特定のkeyでグループ化したData"""
        ...

    def p_choice(self, question_id: str) -> Dict:
        """(choice, p_choice)の辞書"""
        ...


@dataclass
class Data:
    question_id: str
    case_id: str
    p_case: float
    yes: float
    probably_yes: float
    dont_know: float
    probably_no: float
    no: float


class PandasNode(Node):
    def __init__(self, data_list: List[Data]):
        self.df = pd.DataFrame(data_list)

    def groupby(self, by: str) -> Iterable[Tuple[str, Iterable, Iterable]]:
        for question_id, group in self.df.groupby(by):
            yield str(question_id), group, self.df.drop(group.index)

    def p_choice(self, question_id: str) -> Dict:
        """(choice, p_choice)の辞書"""
        group = self.df[self.df['question_id'] == question_id]
        weighted_ps = group[[
            'p_case', 'yes', 'probably_yes', 'dont_know', 'probably_no', 'no']].copy()
        for col in ['yes', 'probably_yes', 'dont_know', 'probably_no', 'no']:
            weighted_ps[col] = weighted_ps[col] * \
                weighted_ps['p_case']
        summed_ps = weighted_ps[['yes', 'probably_yes',
                                 'dont_know', 'probably_no', 'no']].sum()
        total_sum = summed_ps.sum()
        normalized_ps = summed_ps / total_sum
        return normalized_ps.to_dict()


class QuestionSelector:
    def __init__(self, context: Node):
        self.context = context
        self.p_choices = {}

    def select(self):
        for question_id, group, other in self.context.groupby('question_id'):
            self.p_choices = self.context.p_choice(question_id)
            joint_ent = self.entropy(question_id, group, other)
            print(f"question_id: {question_id}, ent: {joint_ent}")

    def entropy(self, question_id, group, other):
        total_ent = 0
        for choice, p in self.p_choices.items():
            ent = self.entropy_per_choice(choice, question_id, group, other)
            total_ent += p * ent
        return total_ent

    def entropy_per_choice(self, choice, question_id, group, other):
        p_choice = self.p_choices[choice]
        p_dist = {}  # case_idとchoiceの２つのindexを持つ
        p_undefined_choice = 0
        p_in_group = 0

        for _, qcc in group.iterrows():
            p_case = qcc['p_case']
            p_choice_case = qcc[choice]
            p_case_choice = (p_choice_case * p_case) / p_choice
            p_dist[qcc['case_id']] = {choice: p_case_choice}
            p_undefined_choice += p_choice_case * p_case
            p_in_group += p_case

        # 質問が未定義の場合の選択肢の確率分布は質問が定義済みのすべての場合から集計される期待値を正規化して推定する
        p_undefined_choice /= p_in_group

        # case_idがgroupにない場合は、otherから取得する
        for case_id, group in other.groupby('case_id'):
            if case_id in p_dist:
                continue
            p_case = group.head(1)['p_case'].iloc[0]
            p_case_choice = (p_undefined_choice * p_case) / p_choice
            p_dist[case_id] = {choice: p_case_choice}

        ent = -sum([p[choice] * math.log2(p[choice]) if p[choice]
                   > 0 else 0 for p in p_dist.values()])
        print(p_dist)
        return ent


data = [
    # is_it_deliciousは、果物を特定する上でヒューリスティックに考えてあまり質問の意味がない質問
    {"case_id": "apple", "p_case": 0.5, "question_id": "is_it_delicious", "yes": 0.8,
        "probably_yes": 0.15, "dont_know": 0.03, "probably_no": 0.02, "no": 0.0},

    {"case_id": "banana", "p_case": 0.4, "question_id": "is_it_delicious", "yes": 0.7,
        "probably_yes": 0.2, "dont_know": 0.05, "probably_no": 0.04, "no": 0.01},

    {"case_id": "watermelon", "p_case": 0.1, "question_id": "is_it_delicious",
        "yes": 0.6, "probably_yes": 0.2, "dont_know": 0.1, "probably_no": 0.08, "no": 0.02},

    # is_it_heavyは、ヒューリスティックに考えてスイカかそれ以外かわかるため意味のある質問
    {"case_id": "apple", "p_case": 0.5, "question_id": "is_it_heavy", "yes": 0.2,
        "probably_yes": 0.3, "dont_know": 0.4, "probably_no": 0.05, "no": 0.05},

    {"case_id": "banana", "p_case": 0.4, "question_id": "is_it_heavy", "yes": 0.1,
        "probably_yes": 0.2, "dont_know": 0.5, "probably_no": 0.15, "no": 0.05},

    {"case_id": "watermelon", "p_case": 0.1, "question_id": "is_it_heavy", "yes": 0.9,
        "probably_yes": 0.08, "dont_know": 0.01, "probably_no": 0.01, "no": 0.0},

    # is_it_animalは、どの場合でも同様にnoになる質問であり、最も意味のない質問、未定義の質問の確率分布は定義済みの確率分布の期待値となり、今回の場合は同じになる
    {"case_id": "apple", "p_case": 0.5, "question_id": "is_it_animal", "yes": 0.01,
        "probably_yes": 0.01, "dont_know": 0.01, "probably_no": 0.01, "no": 0.96},

    # is_it_similar_to_apple_than_bananaは、appleの場合にyes, bananaの場合にno, watermelonの場合にdont_knowになり、最も意味のある質問
    {"case_id": "apple", "p_case": 0.5, "question_id": "is_it_similar_to_apple_than_banana", "yes": 0.6,
        "probably_yes": 0.25, "dont_know": 0.05, "probably_no": 0.05, "no": 0.05},

    {"case_id": "banana", "p_case": 0.4, "question_id": "is_it_similar_to_apple_than_banana", "yes": 0.05,
        "probably_yes": 0.05, "dont_know": 0.05, "probably_no": 0.25, "no": 0.6},

    {"case_id": "watermelon", "p_case": 0.1, "question_id": "is_it_similar_to_apple_than_banana", "yes": 0.05,
        "probably_yes": 0.05, "dont_know": 0.8, "probably_no": 0.05, "no": 0.05},
]


# クラスのインスタンス化とselectメソッドの呼び出し
selector = QuestionSelector(PandasNode(data))  # type: ignore
selector.select()
