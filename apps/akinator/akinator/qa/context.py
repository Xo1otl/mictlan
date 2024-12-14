from typing import Dict, TypedDict
import torch


class ChoiceProbabilities(TypedDict):
    yes: float
    probably_yes: float
    dont_know: float
    probably_no: float
    no: float


class Dataset(TypedDict):
    p_case: Dict[str, float]
    p_choice_given_case_question: Dict[str, Dict[str, ChoiceProbabilities]]


class Probabilities(TypedDict):
    p_choice_given_case_question: float
    p_choice_case_given_question: float
    p_choice_given_question: float
    p_case_given_choice_question: float


class Context:
    def __init__(self, dataset: Dataset, device: torch.device):
        self.question_idx_to_id: Dict[int, str] = {}
        self.question_id_to_idx: Dict[str, int] = {}
        self.case_id_to_idx: Dict[str, int] = {}
        self.case_idx_to_id: Dict[int, str] = {}
        self.choice_to_idx: Dict[str, int] = {
            choice: idx for idx, choice in enumerate((ChoiceProbabilities.__annotations__).keys())}
        self.probability_attr_to_idx = {
            attr: idx for idx, attr in enumerate(Probabilities.__annotations__.keys())}
        self.p_case_tensor = torch.tensor(
            [dataset["p_case"][case_id] for case_id in dataset["p_case"]], device=device)

        num_questions = len(dataset["p_choice_given_case_question"])
        num_cases = len(dataset["p_case"])

        self.tensor = torch.full((
            num_questions,
            num_cases,
            len(self.choice_to_idx),
            len(self.probability_attr_to_idx)),
            float('nan'), device=device)

        # 三重forloopを回してデータが存在する組み合わせに対して値を代入
        question_idx = 0
        case_idx = 0
        for question_id, case_data in dataset["p_choice_given_case_question"].items():
            self.question_idx_to_id[question_idx] = question_id
            self.question_id_to_idx[question_id] = question_idx
            for case_id, choice_data in case_data.items():
                if case_id not in self.case_id_to_idx:
                    self.case_id_to_idx[case_id] = case_idx
                    self.case_idx_to_id[case_idx] = case_id
                    case_idx += 1
                for choice_name, p_choice_given_case_question in choice_data.items():
                    self.tensor[
                        question_idx,
                        self.case_id_to_idx[case_id],
                        self.choice_to_idx[choice_name],
                        self.probability_attr_to_idx["p_choice_given_case_question"]] = p_choice_given_case_question  # type: ignore
            question_idx += 1
        return

    def complete(self):
        p_choice_case_given_question = self.tensor[
            :, :, :, self.probability_attr_to_idx["p_choice_given_case_question"]] * self.p_case_tensor.unsqueeze(0).unsqueeze(2)

        # Calculate normalized p_choice_given_question
        p_choice_given_question = p_choice_case_given_question.nansum(
            dim=1, keepdim=True)
        p_choice_given_question = p_choice_given_question / \
            p_choice_given_question.sum(dim=2, keepdim=True)

        # 欠損値をp_choice_given_questionで補完
        nan_mask = torch.isnan(
            self.tensor[:, :, :, self.probability_attr_to_idx["p_choice_given_case_question"]])
        self.tensor[:, :, :, self.probability_attr_to_idx["p_choice_given_case_question"]] = torch.where(
            nan_mask,
            p_choice_given_question,
            self.tensor[
                :, :, :, self.probability_attr_to_idx["p_choice_given_case_question"]]
        )
