from typing import List, Dict, Tuple, TypedDict
import pandas as pd
import math


class Question(TypedDict):
    id: str
    text: str


class Case(TypedDict):
    id: str
    text: str
    initial_p: float


class QCC(TypedDict):
    case_id: str
    case_p: float
    question_id: str
    yes: float
    probably_yes: float
    dont_know: float
    probably_no: float
    no: float


class QuestionSelector:
    def __init__(self, qccs: List[QCC]):
        self.qccdf = pd.DataFrame(qccs)

    def select(self) -> Tuple[str, float]:
        qid_entropy: Dict[str, float] = {}
        for qid, group in self.qccdf.groupby('question_id'):
            choice_ps = self._choice_ps_per_group(group)
            cc_per_qs = group[[
                'case_id', 'case_p', 'yes', 'probably_yes', 'dont_know', 'probably_no', 'no'
            ]].to_dict(orient='records')
            joint_entropy = Entropy(
                choice_ps, cc_per_qs).posterior()  # type: ignore
            qid_entropy[str(qid)] = joint_entropy
        print(qid_entropy)
        best_qid = min(qid_entropy, key=lambda k: qid_entropy[k])
        return best_qid, qid_entropy[best_qid]

    def _choice_ps_per_group(self, group: pd.DataFrame) -> "ChoicePs":
        weighted_ps = group[[
            'case_p', 'yes', 'probably_yes', 'dont_know', 'probably_no', 'no']].copy()
        for col in ['yes', 'probably_yes', 'dont_know', 'probably_no', 'no']:
            weighted_ps[col] = weighted_ps[col] * \
                weighted_ps['case_p']

        summed_ps = weighted_ps[['yes', 'probably_yes',
                                 'dont_know', 'probably_no', 'no']].sum()

        return ChoicePs(
            yes=summed_ps['yes'],
            probably_yes=summed_ps['probably_yes'],
            dont_know=summed_ps['dont_know'],
            probably_no=summed_ps['probably_no'],
            no=summed_ps['no'],
        )


class ChoicePs(TypedDict):
    yes: float
    probably_yes: float
    dont_know: float
    probably_no: float
    no: float


class CCPerQ(TypedDict):
    case_id: str
    case_p: float
    yes: float
    probably_yes: float
    dont_know: float
    probably_no: float
    no: float


class Entropy:
    def __init__(self, choice_ps: ChoicePs, cc_per_qs: List[CCPerQ]):
        self.choice_ps = choice_ps
        self.cc_per_qs = cc_per_qs

    def _per_choice_posterior(self, choice: str) -> float:
        choice_p = self.choice_ps[choice]
        ps = []
        for ccset in self.cc_per_qs:
            case_p = ccset['case_p']
            cond_p = ccset[choice]
            p = (cond_p * case_p) / choice_p
            ps.append(p)

        posterior_entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in ps)
        return posterior_entropy

    def posterior(self) -> float:
        total_posterior_entropy = 0.0
        for choice, p in self.choice_ps.items():
            posterior_ent = self._per_choice_posterior(choice)
            total_posterior_entropy += p * posterior_ent  # type: ignore
        return total_posterior_entropy


qccs = [
    # is_it_deliciousは、果物を特定する上でヒューリスティックに考えてあまり質問の意味がない質問
    {"case_id": "apple", "case_p": 0.4, "question_id": "is_it_delicious", "yes": 0.8,
        "probably_yes": 0.15, "dont_know": 0.03, "probably_no": 0.02, "no": 0.0},

    {"case_id": "banana", "case_p": 0.3, "question_id": "is_it_delicious", "yes": 0.7,
        "probably_yes": 0.2, "dont_know": 0.05, "probably_no": 0.04, "no": 0.01},

    {"case_id": "watermelon", "case_p": 0.3, "question_id": "is_it_delicious",
        "yes": 0.6, "probably_yes": 0.2, "dont_know": 0.1, "probably_no": 0.08, "no": 0.02},

    # is_it_heavyは、ヒューリスティックに考えてスイカかそれ以外かわかるため意味のある質問
    {"case_id": "apple", "case_p": 0.4, "question_id": "is_it_heavy", "yes": 0.2,
        "probably_yes": 0.3, "dont_know": 0.4, "probably_no": 0.05, "no": 0.05},

    {"case_id": "banana", "case_p": 0.3, "question_id": "is_it_heavy", "yes": 0.1,
        "probably_yes": 0.2, "dont_know": 0.5, "probably_no": 0.15, "no": 0.05},

    {"case_id": "watermelon", "case_p": 0.3, "question_id": "is_it_heavy", "yes": 0.9,
        "probably_yes": 0.08, "dont_know": 0.01, "probably_no": 0.01, "no": 0.0},

    # is_it_animalは、どの場合でも同様にnoになる質問であり、最も意味のない質問
    {"case_id": "apple", "case_p": 0.4, "question_id": "is_it_animal", "yes": 0.01,
        "probably_yes": 0.01, "dont_know": 0.01, "probably_no": 0.01, "no": 0.96},

    {"case_id": "banana", "case_p": 0.3, "question_id": "is_it_animal", "yes": 0.01,
        "probably_yes": 0.01, "dont_know": 0.01, "probably_no": 0.01, "no": 0.96},

    {"case_id": "watermelon", "case_p": 0.3, "question_id": "is_it_animal", "yes": 0.01,
        "probably_yes": 0.01, "dont_know": 0.01, "probably_no": 0.01, "no": 0.96},

    # is_it_similar_to_apple_than_bananaは、appleの場合にyes, bananaの場合にno, watermelonの場合にdont_knowになり、最も意味のある質問
    {"case_id": "apple", "case_p": 0.4, "question_id": "is_it_similar_to_apple_than_banana", "yes": 0.6,
        "probably_yes": 0.25, "dont_know": 0.05, "probably_no": 0.05, "no": 0.05},

    {"case_id": "banana", "case_p": 0.3, "question_id": "is_it_similar_to_apple_than_banana", "yes": 0.05,
        "probably_yes": 0.05, "dont_know": 0.05, "probably_no": 0.25, "no": 0.6},

    {"case_id": "watermelon", "case_p": 0.3, "question_id": "is_it_similar_to_apple_than_banana", "yes": 0.05,
        "probably_yes": 0.05, "dont_know": 0.8, "probably_no": 0.05, "no": 0.05},
]

selector = QuestionSelector(qccs)  # type: ignore
qid, entropy = selector.select()
print(qid, entropy)
