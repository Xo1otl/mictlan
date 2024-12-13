import json
from typing import Dict, TypedDict, List
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
        self.idx_to_question_id: Dict[int, str] = {}
        self.case_id_map: Dict[str, int] = {}
        self.choice_name_map: Dict[str, int] = {
            attr: idx for idx, attr in enumerate((ChoiceProbabilities.__annotations__).keys())}
        self.context_attr_map = {
            attr: idx for idx, attr in enumerate(Probabilities.__annotations__.keys())}
        self.p_case_tensor = torch.tensor(
            [dataset["p_case"][case_id] for case_id in dataset["p_case"]], device=device)

        num_questions = len(dataset["p_choice_given_case_question"])
        num_cases = len(dataset["p_case"])

        self.tensor = torch.full((
            num_questions,
            num_cases,
            len(self.choice_name_map),
            len(self.context_attr_map)),
            float('nan'), device=device)

        # 三重forloopを回してデータが存在する組み合わせに対して値を代入
        q_idx = 0
        c_idx = 0
        for question_id, case_data in dataset["p_choice_given_case_question"].items():
            self.idx_to_question_id[q_idx] = question_id
            for case_id, choice_data in case_data.items():
                if case_id not in self.case_id_map:
                    self.case_id_map[case_id] = c_idx
                    c_idx += 1
                for choice_name, p_choice_given_case_question in choice_data.items():
                    self.tensor[
                        q_idx,
                        self.case_id_map[case_id],
                        self.choice_name_map[choice_name],
                        self.context_attr_map["p_choice_given_case_question"]] = p_choice_given_case_question  # type: ignore
            q_idx += 1

        return

    def complete(self):
        # p_choice_given_case_questionが存在する場所のマスク
        mask = ~torch.isnan(
            self.tensor[:, :, :, self.context_attr_map["p_choice_given_case_question"]])
        weighted_p_choice = self.tensor[:, :, :, self.context_attr_map["p_choice_given_case_question"]
                                        ] * self.p_case_tensor.unsqueeze(0).unsqueeze(2)
        sum_weights = torch.where(mask, self.p_case_tensor.unsqueeze(
            0).unsqueeze(2), torch.zeros_like(weighted_p_choice))
        sum_weights_per_group = sum_weights.sum(dim=1, keepdim=True)
        # 質問と選択肢ごとの加重平均を計算
        weighted_sum_p_choice = torch.where(
            mask, weighted_p_choice, torch.zeros_like(weighted_p_choice))
        weighted_avg_p_choice = weighted_sum_p_choice.sum(dim=1, keepdim=True) / torch.where(
            sum_weights_per_group > 0, sum_weights_per_group, torch.ones_like(sum_weights_per_group))
        # 欠損値を加重平均で補完
        self.tensor[:, :, :, self.context_attr_map["p_choice_given_case_question"]] = torch.where(
            mask,
            self.tensor[:, :, :,
                        self.context_attr_map["p_choice_given_case_question"]],
            weighted_avg_p_choice.expand_as(
                self.tensor[:, :, :, self.context_attr_map["p_choice_given_case_question"]])
        )


class Selector:
    def __init__(self, conetxt: Context, asked_questions: List[str]):
        self.context = conetxt
        self.asked_questions = asked_questions
        self.best_question_id = None

    def best_question(self):
        entropies = self.question_entropies()
        print(entropies)

        # Filter out entropies for already asked questions
        filtered_entropies = []
        filtered_question_indices = []
        for i, entropy in enumerate(entropies):
            question_id = self.context.idx_to_question_id[i]
            if question_id not in self.asked_questions:
                filtered_entropies.append(entropy)
                filtered_question_indices.append(i)

        if not filtered_entropies:
            # Handle case where all questions have been asked
            print("All questions have been asked.")
            return None  # Or raise an exception if appropriate

        # Find the best question from the filtered list
        best_question_idx_filtered = torch.argmin(
            torch.tensor(filtered_entropies))
        best_question_idx = filtered_question_indices[best_question_idx_filtered]

        self.best_question_id = self.context.idx_to_question_id[int(
            best_question_idx)]
        return self.best_question_id

    def question_entropies(self) -> torch.Tensor:
        # p_choice_case_given_question = p_choice_given_case_question * p_case
        self.context.tensor[:, :, :, self.context.context_attr_map["p_choice_case_given_question"]] = (
            self.context.tensor[
                :, :, :, self.context.context_attr_map["p_choice_given_case_question"]]
            * self.context.p_case_tensor.unsqueeze(0).unsqueeze(2)
        )

        # p_choice_given_question = Σ_case p_choice_case_given_question
        self.context.tensor[:, :, :, self.context.context_attr_map["p_choice_given_question"]] = (
            self.context.tensor[:, :, :, self.context.context_attr_map["p_choice_case_given_question"]].sum(
                dim=1, keepdim=True)
        )

        # p_case_given_choice_question = p_choice_case_given_question / p_choice_given_question
        self.context.tensor[:, :, :, self.context.context_attr_map["p_case_given_choice_question"]] = (
            self.context.tensor[
                :, :, :, self.context.context_attr_map["p_choice_case_given_question"]]
            / self.context.tensor[:, :, :, self.context.context_attr_map["p_choice_given_question"]]
        )

        p_case_given_choice_question = self.context.tensor[
            :, :, :, self.context.context_attr_map["p_case_given_choice_question"]]
        log_p_case_given_choice_question = torch.where(p_case_given_choice_question > 0, torch.log2(
            p_case_given_choice_question), torch.zeros_like(p_case_given_choice_question))
        entropy_for_each_case = -p_case_given_choice_question * \
            log_p_case_given_choice_question

        # 質問ごと、選択肢ごとのエントロピーを計算
        entropy_per_question_choice = entropy_for_each_case.sum(dim=1)
        p_choice_given_question = self.context.tensor[
            :, :, :, self.context.context_attr_map["p_choice_given_question"]][:, 0, :]

        weighted_entropy = entropy_per_question_choice * p_choice_given_question

        # 質問ごとのエントロピーを計算
        question_entropies = weighted_entropy.sum(dim=1)
        return question_entropies

    def update_context(self, question_id: str, choice: str):
        # 情報エントロピーの計算の過程で、context.tensorの中にすでに計算済みの確率分布がある
        # それを取得してcontext.p_case_tensorを更新する
        question_idx = [
            idx for idx, q_id in self.context.idx_to_question_id.items() if q_id == question_id][0]
        choice_idx = self.context.choice_name_map[choice]
        self.context.p_case_tensor = self.context.tensor[
            question_idx, :, choice_idx, self.context.context_attr_map["p_case_given_choice_question"]
        ].clone()  # cloneしないと、tensorの中身が書き換えられたときにp_case_tensorも書き換えられてしまう
        self.asked_questions.append(question_id)


def recursive_ask(context: Context, asked_questions=[], top_n: int = 3):
    """
    Recursively asks questions until a confident answer is reached.

    Args:
        context: The context object containing probability distributions.
        selector: The selector object used to choose the best question.
        top_n: The number of top cases to display.
    """

    # Get the indices of the top n probabilities and their values
    top_probs, top_indices = torch.topk(
        context.p_case_tensor, min(top_n, len(context.p_case_tensor)))

    print("Current most likely cases:")
    for i in range(len(top_probs)):
        case_idx = top_indices[i].item()
        case_prob = top_probs[i].item()
        case_id = [case_id for case_id,
                   idx in context.case_id_map.items() if idx == case_idx][0]
        print(f"  {i+1}. {case_id} ({case_prob:.4f})")

    # Termination conditions (using top 2 for decision logic)
    if len(top_probs) >= 2:
        top_prob = top_probs[0].item()
        second_top_prob = top_probs[1].item()
        if top_prob > 0.5 or top_prob >= 2 * second_top_prob:
            top_case_id = [case_id for case_id, idx in context.case_id_map.items(
            ) if idx == top_indices[0]][0]
            print(
                f"The most likely case is {top_case_id} with probability {top_prob:.4f}.")
            return
    elif len(top_probs) == 1:
        top_prob = top_probs[0].item()
        top_case_id = [case_id for case_id, idx in context.case_id_map.items(
        ) if idx == top_indices[0]][0]
        if top_prob > 0.5:
            print(
                f"The most likely case is {top_case_id} with probability {top_prob:.4f}.")
            return

    selector = Selector(context, asked_questions)
    # Choose the best question
    best_question_id = selector.best_question()
    print(f"Question: {best_question_id}?")

    # Get user input
    while True:
        choice = input(
            "Answer (yes/probably_yes/dont_know/probably_no/no): ").lower()
        if choice in context.choice_name_map:
            break
        print(
            "Invalid choice. Please choose from yes/probably_yes/dont_know/probably_no/no.")

    # Update context
    selector.update_context(str(best_question_id), choice)
    asked_questions.append(best_question_id)

    # Recursive call
    recursive_ask(context, asked_questions, top_n)


# GPUが利用可能か確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

dataset: Dataset = {
    "p_case": {
        "apple": 0.4,
        "banana": 0.4,
        "watermelon": 0.2
    },
    "p_choice_given_case_question": {
        "is_it_delicious": {
            "apple": {"yes": 0.8, "probably_yes": 0.15, "dont_know": 0.03, "probably_no": 0.02, "no": 0.0},
            "banana": {"yes": 0.7, "probably_yes": 0.2, "dont_know": 0.05, "probably_no": 0.04, "no": 0.01},
            "watermelon": {"yes": 0.6, "probably_yes": 0.2, "dont_know": 0.1, "probably_no": 0.08, "no": 0.02}
        },
        "is_it_heavy": {
            "apple": {"yes": 0.2, "probably_yes": 0.3, "dont_know": 0.4, "probably_no": 0.05, "no": 0.05},
            "banana": {"yes": 0.1, "probably_yes": 0.2, "dont_know": 0.5, "probably_no": 0.15, "no": 0.05},
            "watermelon": {"yes": 0.9, "probably_yes": 0.08, "dont_know": 0.01, "probably_no": 0.01, "no": 0.0}
        },
        "is_it_animal": {
            "apple": {"yes": 0.01, "probably_yes": 0.01, "dont_know": 0.01, "probably_no": 0.01, "no": 0.96},
            # "banana" に関する行自体が存在しない
            # "watermelon" に関する行自体が存在しない
        },
        "is_it_similar_to_apple_than_watermelon": {
            "apple": {"yes": 0.9, "probably_yes": 0.1, "dont_know": 0.0, "probably_no": 0.0, "no": 0.0},
            "banana": {"yes": 0.01, "probably_yes": 0.01, "dont_know": 0.96, "probably_no": 0.01, "no": 0.01},
            "watermelon": {"yes": 0.0, "probably_yes": 0.0, "dont_know": 0.0, "probably_no": 0.1, "no": 0.9}
        }
    }
}

dataset = json.loads(
    open("/workspaces/mictlan/apps/akinator/akinator/qa/output_transformed.json").read())
context = Context(dataset, device)
context.complete()  # completeメソッドを呼び出して補完を実行

print("テンソルの形状:", context.tensor.shape)
print("テンソルの次元数:", context.tensor.ndim)
print("テンソルの要素数:", context.tensor.numel())

selector = Selector(context, [])
# best_question_id = selector.best_question()
# print(f"最適な質問: {best_question_id}")

recursive_ask(context, top_n=5)
