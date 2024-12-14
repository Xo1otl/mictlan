import json
from typing import Dict, List, TypedDict
import pandas as pd
import numpy as np
import torch


def calc_completed_choice_df(data) -> pd.DataFrame:
    # --- DataFrameに変換 ---
    df_list = []
    for question_id, case_data in data["p_choice_given_case_question"].items():
        for case_id, choice_data in case_data.items():
            for choice, probability in choice_data.items():
                df_list.append([question_id, case_id, choice, probability])

    df = pd.DataFrame(df_list, columns=[
                      "question_id", "case_id", "choice", "p_choice_given_case_question"])

    # --- 欠損値を補完するために、まず全ての組み合わせを生成 ---
    # すべての question_id, case_id, choice の組み合わせを取得
    all_questions = df["question_id"].unique()
    all_cases = df["case_id"].unique()
    all_choices = df["choice"].unique()

    # すべての組み合わせのDataFrameを作成
    all_combinations = pd.MultiIndex.from_product(
        [all_questions, all_cases, all_choices],
        names=["question_id", "case_id", "choice"]
    ).to_frame(index=False)

    # 元のDataFrameと結合（存在しない組み合わせにはNaNが入る）
    df = pd.merge(all_combinations, df, on=[
                  "question_id", "case_id", "choice"], how="left")

    # --- 欠損値補完 ---
    # p_caseのDataFrameも作成
    df_p_case = pd.DataFrame(list(data["p_case"].items()), columns=[
                             "case_id", "p_case"])

    # 加重平均を計算するための関数
    def weighted_mean(group, df_p_case):
        question_id, choice = group.name

        # 該当する question_id と choice を持つ行をフィルタリング
        available_data = df[(df["question_id"] == question_id) & (
            df["choice"] == choice) & (~df["p_choice_given_case_question"].isna())]

        # 重みを取得するために、case_id を使って df_p_case と結合
        weighted_data = pd.merge(available_data, df_p_case, on="case_id")

        if weighted_data.empty:
            # データがない場合は NaN を返す
            return pd.Series([np.nan] * len(group), index=group.index)

        # 重み付き平均を計算
        result = np.average(
            weighted_data["p_choice_given_case_question"], weights=weighted_data["p_case"])
        return pd.Series([result] * len(group), index=group.index)

    # 欠損値を補完
    df["p_choice_given_case_question"] = df.groupby(["question_id", "choice"], group_keys=False)[
        "p_choice_given_case_question"].apply(lambda group: group.fillna(weighted_mean(group, df_p_case)))

    return df


class Selector:
    """
    質問選択と確率更新のインターフェースを定義するプロトコル。
    """

    def __init__(self, completed_choice_df: pd.DataFrame, p_case: Dict[str, float], asked_questions: List[str] = []):
        """
        コンストラクタ。

        Args:
            completed_choice_df_df: 補完された p_choice_given_case_question を含むデータフレーム。
            p_case: 現在の確率分布 (各場合の確率)。
        """
        self.completed_choice_df = completed_choice_df
        self.df_p_case = pd.DataFrame(
            list(p_case.items()), columns=["case_id", "p_case"])
        self.best_question_id = None
        self.df_joined = None
        self.asked_questions = asked_questions

    def select_best_question(self) -> str:
        """
        現在の p_case に基づいて、情報利得が最大となる質問を選択する。

        Returns:
            選択された質問のID (question_id)。
        """
        # --- 最もエントロピーが小さい質問を選択 ---
        question_entropy = self._question_entropy()
        print(question_entropy)
        best_question = question_entropy.loc[question_entropy["entropy"].idxmin(
        )]
        self.best_question_id = best_question['question_id']
        return str(self.best_question_id)

    def update_p_case(self, question_id: str, choice: str) -> Dict[str, float]:
        """
        ユーザーが選択した質問と回答に基づいて、p_case を更新する。

        Args:
            question_id: 選択された質問のID。
            choice: ユーザーが選択した選択肢。

        Returns:
            更新された p_case (新しい確率分布)。
        """
        if self.df_joined is None:
            raise ValueError("select_best_question() を先に実行してください。")

        # --- 事後確率の計算 ---
        # df_joined から、選択された質問と回答に対応する p_case_given_choice_question を取得
        posterior_probs = self.df_joined.loc[(question_id, slice(
            None), choice), "p_case_given_choice_question"]  # type: ignore

        # インデックスをリセットして、case_id を列に戻す
        posterior_probs = posterior_probs.reset_index(level="case_id")

        # case_id ごとの確率を取得
        posterior_probs = posterior_probs.set_index("case_id")[
            "p_case_given_choice_question"]

        # 事後確率を正規化して、合計が 1 になるようにする
        posterior_probs = posterior_probs / posterior_probs.sum()
        self.p_case = posterior_probs.to_dict()

        return self.p_case

    def _question_entropy(self) -> pd.DataFrame:
        """
        現在の p_case に基づいて、エントロピーを計算する。

        Returns:
            計算されたエントロピーを含むデータフレーム。
        """
        df_unasked = self.completed_choice_df[~self.completed_choice_df["question_id"].isin(
            self.asked_questions)]

        # マルチインデックスを設定
        df_filled = df_unasked.set_index(
            ["question_id", "case_id", "choice"])

        # p_caseのDataFrameと結合
        df_joined = df_filled.join(self.df_p_case.set_index(
            "case_id"), on="case_id")

        # p_choice_given_question_case * p_case を計算
        df_joined["p_choice_case_given_question"] = (
            df_joined["p_choice_given_case_question"] * df_joined["p_case"]
        )

        # question と choice で集約して p_choice_given_question を計算
        p_choice_given_question = df_joined.groupby(["question_id", "choice"])[
            "p_choice_case_given_question"
        ].sum()
        p_choice_given_question.name = "p_choice_given_question"

        # p_choice_given_question_case_times_p_case と p_choice_given_question を結合
        df_joined = df_joined.join(p_choice_given_question, on=[
                                   "question_id", "choice"])

        # p_case_given_choice_question を計算
        df_joined["p_case_given_choice_question"] = (
            df_joined["p_choice_case_given_question"] /
            df_joined["p_choice_given_question"]
        )
        self.df_joined = df_joined

        # 質問と選択肢でグループ化し、各グループ(場合)のエントロピーを計算
        entropy_df = df_joined.groupby(["question_id", "choice"])[
            "p_case_given_choice_question"
        ].apply(self.calculate_entropy).reset_index()

        # p_choice_given_question を entropy_df に結合
        entropy_df = entropy_df.merge(
            p_choice_given_question, on=["question_id", "choice"]
        )

        # 各質問・選択肢のエントロピーと選択肢の出現確率を掛け合わせる
        entropy_df["weighted_entropy"] = (
            entropy_df["p_case_given_choice_question"] *
            entropy_df["p_choice_given_question"]
        )

        # 質問ごとに、重み付けされたエントロピーの合計を計算
        entropy_df = entropy_df.groupby("question_id")[
            "weighted_entropy"].sum().reset_index()

        entropy_df.rename(
            columns={"weighted_entropy": "entropy"}, inplace=True)
        return entropy_df

    def calculate_entropy(self, probabilities):
        return -np.sum(np.where(probabilities > 0, probabilities * np.log2(probabilities), 0))


data = json.loads(
    open("/workspaces/mictlan/apps/akinator/akinator/qa/output_transformed.json").read())

completed_choice_df = calc_completed_choice_df(data)


def recursive_question_selection(completed_choice_df, p_case, asked_questions=[]):
    """
    再帰的に質問を選択し、確率を更新する関数

    Args:
        completed_choice_df: 完了した選択肢のDataFrame
        p_case: 各ケースの確率 (Dict[str, float])
        selector_class: Selectorクラス (ここでは引数で渡すように変更)

    Returns:
        None (確率が0.5以上のケースが見つかった場合、そのケースを表示して終了)
    """

    max_case = max(p_case, key=p_case.get)
    max_prob = p_case[max_case]
    top_5_cases = sorted(
        p_case.items(), key=lambda item: item[1], reverse=True)[:5]
    print("確率の上位5ケース:")
    for case, prob in top_5_cases:
        print(f"  ケース {case}: {prob}")

    if max_prob >= 0.5:
        print(f"ケース {max_case} の確率が0.5以上になりました: {max_prob}")
        return

    if max_prob >= 2 * top_5_cases[1][1]:
        print(
            f"ケース {max_case} の確率が2位の2倍以上になりました: {max_prob} >= {2 * top_5_cases[1][1]}")
        return True, max_case

    # Selectorオブジェクトの作成
    selector = Selector(completed_choice_df, p_case, asked_questions)

    # 最適な質問を選択
    question_id = selector.select_best_question()
    asked_questions.append(question_id)
    if question_id is None:
        print("質問が残っていません。")
        return  # 質問がない場合は終了

    # ユーザーに回答を求める
    print(f"質問ID {question_id} に対する回答を選択してください:")
    print("1: yes")
    print("2: probably_yes")
    print("3: dont_know")
    print("4: probably_no")
    print("5: no")

    while True:
        try:
            user_input = int(input("選択肢 (1-5): "))
            if 1 <= user_input <= 5:
                break
            else:
                print("1から5の数字で入力してください。")
        except ValueError:
            print("数字で入力してください。")

    # 回答を文字列に変換
    answers = {
        1: "yes",
        2: "probably_yes",
        3: "dont_know",
        4: "probably_no",
        5: "no"
    }
    answer = answers[user_input]

    # 確率を更新
    p_case = selector.update_p_case(question_id, answer)

    # 再帰呼び出し
    recursive_question_selection(completed_choice_df, p_case, asked_questions)


recursive_question_selection(completed_choice_df, data["p_case"])
