from typing import Protocol
import pandas as pd


class Context(Protocol):
    def groupby(self, by: str):
        """特定のkeyでグループ化したData"""
        ...

    def p_cases(self):
        """(case_id, p_case)のリスト"""
        ...

    def p_choice(self, question_id: str):
        """(choice, p_choice)のリスト"""
        ...


class QestionSelector:
    def __init__(self, qccs: pd.DataFrame):
        self.qccs = qccs
        self.choice_ps = pd.DataFrame({})

    def select(self):
        for question_id, group in self.qccs.groupby('question_id'):
            self.choice_ps = self._choice_ps_per_group(group)
            other = self.qccs.drop(group.index)
            # other.groupby('case_id')
            joint_ent = self.entropy(question_id, group, other)
            print(joint_ent)

    def _choice_ps_per_group(self, group: pd.DataFrame):
        weighted_ps = group[[
            'case_p', 'yes', 'probably_yes', 'dont_know', 'probably_no', 'no']].copy()
        for col in ['yes', 'probably_yes', 'dont_know', 'probably_no', 'no']:
            weighted_ps[col] = weighted_ps[col] * \
                weighted_ps['case_p']
        summed_ps = weighted_ps[['yes', 'probably_yes',
                                 'dont_know', 'probably_no', 'no']].sum()
        return summed_ps

    def entropy(self, question_id, group, other):
        total_ent = 0
        for choice, p in self.choice_ps.items():
            ent = self.entropy_per_choice(choice, question_id, group, other)
            total_ent += p * ent  # type: ignore
        return total_ent

    def entropy_per_choice(self, choice, question_id, group, other):
        choice_p = self.choice_ps[choice]
        dist = []
        expected_p = 0
        in_group_p = 0

        for _, qcc in group.iterrows():
            case_p = qcc['case_p']
            p_choice_case = qcc[choice]
            p_case_choice = (p_choice_case * case_p) / choice_p
            dist.append(
                (question_id, qcc['case_id'], choice, p_case_choice))
            expected_p += p_choice_case * case_p
            in_group_p += case_p

        expected_p /= in_group_p

        for _, qcc in other.iterrows():
            case_p = qcc['case_p']
            p_case_choice = (expected_p * case_p) / choice_p
            dist.append(
                (question_id, qcc['case_id'], choice, p_case_choice))

        return 1


data = [
    # is_it_deliciousは、果物を特定する上でヒューリスティックに考えてあまり質問の意味がない質問
    # {"case_id": "apple", "case_p": 0.5, "question_id": "is_it_delicious", "yes": 0.8,
    #     "probably_yes": 0.15, "dont_know": 0.03, "probably_no": 0.02, "no": 0.0},

    {"case_id": "banana", "case_p": 0.4, "question_id": "is_it_delicious", "yes": 0.7,
        "probably_yes": 0.2, "dont_know": 0.05, "probably_no": 0.04, "no": 0.01},

    {"case_id": "watermelon", "case_p": 0.1, "question_id": "is_it_delicious",
        "yes": 0.6, "probably_yes": 0.2, "dont_know": 0.1, "probably_no": 0.08, "no": 0.02},

    # is_it_heavyは、ヒューリスティックに考えてスイカかそれ以外かわかるため意味のある質問
    {"case_id": "apple", "case_p": 0.5, "question_id": "is_it_heavy", "yes": 0.2,
        "probably_yes": 0.3, "dont_know": 0.4, "probably_no": 0.05, "no": 0.05},

    # {"case_id": "banana", "case_p": 0.4, "question_id": "is_it_heavy", "yes": 0.1,
    #     "probably_yes": 0.2, "dont_know": 0.5, "probably_no": 0.15, "no": 0.05},

    {"case_id": "watermelon", "case_p": 0.1, "question_id": "is_it_heavy", "yes": 0.9,
        "probably_yes": 0.08, "dont_know": 0.01, "probably_no": 0.01, "no": 0.0},

    # is_it_animalは、どの場合でも同様にnoになる質問であり、最も意味のない質問
    {"case_id": "apple", "case_p": 0.5, "question_id": "is_it_animal", "yes": 0.01,
        "probably_yes": 0.01, "dont_know": 0.01, "probably_no": 0.01, "no": 0.96},

    {"case_id": "banana", "case_p": 0.4, "question_id": "is_it_animal", "yes": 0.01,
        "probably_yes": 0.01, "dont_know": 0.01, "probably_no": 0.01, "no": 0.96},

    # {"case_id": "watermelon", "case_p": 0.1, "question_id": "is_it_animal", "yes": 0.01,
    #     "probably_yes": 0.01, "dont_know": 0.01, "probably_no": 0.01, "no": 0.96},

    # is_it_similar_to_apple_than_bananaは、appleの場合にyes, bananaの場合にno, watermelonの場合にdont_knowになり、最も意味のある質問
    {"case_id": "apple", "case_p": 0.5, "question_id": "is_it_similar_to_apple_than_banana", "yes": 0.6,
        "probably_yes": 0.25, "dont_know": 0.05, "probably_no": 0.05, "no": 0.05},

    {"case_id": "banana", "case_p": 0.4, "question_id": "is_it_similar_to_apple_than_banana", "yes": 0.05,
        "probably_yes": 0.05, "dont_know": 0.05, "probably_no": 0.25, "no": 0.6},

    # {"case_id": "watermelon", "case_p": 0.1, "question_id": "is_it_similar_to_apple_than_banana", "yes": 0.05,
    #     "probably_yes": 0.05, "dont_know": 0.8, "probably_no": 0.05, "no": 0.05},
]


qccs = pd.DataFrame(data)

# クラスのインスタンス化とselectメソッドの呼び出し
selector = QestionSelector(qccs)
selector.select()
