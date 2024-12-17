import torch
from . import Context
from typing import List


class Selector:
    def __init__(self, conetxt: Context):
        self.context = conetxt
        self.asked_questions: List[int] = []

    def best_question(self):
        entropies = self.question_entropies()
        # 事後分布は既出の確率からは独立している
        # 影響のある選択肢だけを考慮する
        mask = torch.zeros_like(entropies, dtype=torch.bool)
        mask[self.asked_questions] = True
        filtered_entropies = entropies.masked_fill(mask, float('inf'))
        best_question_idx = torch.argmin(filtered_entropies)
        return self.context.question_idx_to_id[int(best_question_idx)]

    @torch.no_grad()
    def question_entropies(self) -> torch.Tensor:
        # P(選択肢,場合|質問) = P(選択肢|場合,質問) * P(場合)
        self.context.tensor[:, :, :, self.context.probability_attr_to_idx["p_choice_case_given_question"]] = (
            self.context.tensor[
                :, :, :, self.context.probability_attr_to_idx["p_choice_given_case_question"]]
            * self.context.p_case_tensor.unsqueeze(0).unsqueeze(2)
        )

        # P(選択肢|質問) = Σ_場合 P(選択肢,場合|質問)
        self.context.tensor[:, :, :, self.context.probability_attr_to_idx["p_choice_given_question"]] = (
            self.context.tensor[:, :, :, self.context.probability_attr_to_idx["p_choice_case_given_question"]].sum(
                dim=1, keepdim=True)
        )

        # P(場合|選択肢,質問) = P(選択肢,場合|質問) / P(選択肢|質問) (ベイズの定理より)
        self.context.tensor[:, :, :, self.context.probability_attr_to_idx["p_case_given_choice_question"]] = (
            self.context.tensor[
                :, :, :, self.context.probability_attr_to_idx["p_choice_case_given_question"]]
            / self.context.tensor[:, :, :, self.context.probability_attr_to_idx["p_choice_given_question"]]
        )

        # エントロピーの計算に必要な条件付き確率等をテンソルから取得
        p_case_given_choice_question = self.context.tensor[
            :, :, :, self.context.probability_attr_to_idx["p_case_given_choice_question"]]
        log_p_case_given_choice_question = torch.where(p_case_given_choice_question > 0, torch.log2(
            p_case_given_choice_question), torch.zeros_like(p_case_given_choice_question))

        # 存在しない組み合わせの項はベイズの定理で0除算され値がnanになるのでnansumを使って除外する
        # H(選択肢,質問) = - Σ_場合 P(場合|選択肢,質問) * log2(P(場合|選択肢,質問))
        entropy_given_choice_question = (
            -p_case_given_choice_question * log_p_case_given_choice_question).nansum(dim=1)
        # H(質問) = Σ_選択肢 H(選択肢,質問) * P(選択肢|質問)
        entropy_given_question = (entropy_given_choice_question * self.context.tensor[
            :, :, :, self.context.probability_attr_to_idx["p_choice_given_question"]][:, 0, :]).nansum(dim=1)
        return entropy_given_question

    @torch.no_grad()
    def update_context(self, question_id: str, choice: str):
        # 同じ質問に対して違う選択肢を選ぶことは想定されていないため、p_case_tensorがすべてnanになる
        # 情報エントロピーの計算の過程で、context.tensorの中にすでに計算済みの確率分布がある
        question_idx = self.context.question_id_to_idx[question_id]
        choice_idx = self.context.choice_to_idx[choice]
        self.context.p_case_tensor = self.context.tensor[
            question_idx, :, choice_idx, self.context.probability_attr_to_idx["p_case_given_choice_question"]
        ].clone()  # cloneしないと、tensorの中身が書き換えられたときにp_case_tensorも書き換えられてしまう

        # 質問と選択肢がわかった時、結果がわかっていなかった条件での選択肢分布は適用できなくなる
        # P(選択肢|質問)はクロネッカーのデルタになると言える
        # すべてのP(選択肢|場合,質問)も同時にクロネッカーのデルタになることが導かれる
        # P(選択肢|質問)がある選択肢で1の時、聞くまでもなくどの結果が得られるかわかっている状態なので、質問する意味がない
        # よって、そのような質問をcontext.tensorから除外したい
        # しかし、idxのマッピングを更新する必要がある
        # id_idxのマップをtensorにしてもいいかもしれない
        # それかquestion_entropy()の中でmaskしてcopyを作りその中で計算を行う
        # それかbest_question()で結果から除外する

        # 一番簡単なbest_question()での除外
        self.asked_questions.append(question_idx)
