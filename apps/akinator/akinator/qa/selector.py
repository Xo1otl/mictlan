import json
from typing import Dict
import pandas as pd
import numpy as np
from treelib import Tree
import random
import string


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

    def __init__(self, completed_choice_df: pd.DataFrame, p_case: Dict[str, float]):
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

    def select_best_question(self) -> str:
        """
        現在の p_case に基づいて、情報利得が最大となる質問を選択する。

        Returns:
            選択された質問のID (question_id)。
        """
        # --- 最もエントロピーが小さい質問を選択 ---
        question_entropy = self._question_entropy()
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

        # マルチインデックスを設定
        df_filled = self.completed_choice_df.set_index(
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
        # エントロピーを計算する関数

    def calculate_entropy(self, probabilities):
        return -np.sum(np.where(probabilities > 0, probabilities * np.log2(probabilities), 0))


data = {
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


def grow_tree(selector: Selector, p_case: Dict[str, float], max_depth: int) -> Tree:
    """
    決定木を成長させる関数。treelib の Tree を直接操作する。

    Args:
        selector: Selector オブジェクト。
        p_case: 初期の確率分布。
        max_depth: 木の最大深さ。

    Returns:
        treelib の Tree オブジェクト。
    """
    tree = Tree()
    # ルートノードを作成
    tree.create_node(
        "Root",  # tag
        "root",  # identifier
        data={"p_case": p_case, "depth": 0,
              "question_id": "root_question"}  # data
    )

    # 再帰的にノードを追加する関数
    def add_node(parent_id: str, depth: int, current_p_case: Dict[str, float], choice=""):
        if depth >= max_depth or any(p >= 0.5 for p in current_p_case.values()):
            # 葉ノードの場合
            tree.create_node(
                f"Leaf: {', '.join(f'{k}: {v:.2f}' for k, v in current_p_case.items())}",
                parent=parent_id,
                data={"p_case": current_p_case,
                      "depth": depth, "is_leaf": True, "question_id": "End"}
            )
            return

        question_id = selector.select_best_question()
        # 質問ノードの場合
        # ノードIDを生成
        node_id = f"{question_id}_{choice}_{''.join(random.choice(string.ascii_letters) for i in range(4))}"
        print(tree)
        tree.create_node(
            f"Q: {question_id} (d: {depth})",
            node_id,
            parent=parent_id,
            data={"p_case": current_p_case, "depth": depth,
                  "question_id": question_id, "is_leaf": False}
        )

        available_choices = selector.completed_choice_df[selector.completed_choice_df[
            "question_id"] == question_id]["choice"].unique()
        for choice in available_choices:
            selector.df_p_case = pd.DataFrame(
                list(current_p_case.items()), columns=["case_id", "p_case"])
            selector._question_entropy()
            updated_p_case = selector.update_p_case(question_id, choice)
            # 子ノードを追加（再帰呼び出し）
            add_node(node_id, depth + 1, updated_p_case, choice)

    # ルートノードから再帰的にノードを追加
    add_node("root", 0, p_case)
    return tree


data = json.loads(
    open("/workspaces/mictlan/apps/akinator/akinator/qa/output_transformed.json").read())

completed_choice_df = calc_completed_choice_df(data)


def recursive_question_selection(completed_choice_df, p_case):
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
    if max_prob >= 0.5:
        print(f"ケース {max_case} の確率が0.5以上になりました: {max_prob}")
        # 確率の高い順にソートして上位5ケースを取得
        top_5_cases = sorted(
            p_case.items(), key=lambda item: item[1], reverse=True)[:5]
        print("確率の上位5ケース:")
        for case, prob in top_5_cases:
            print(f"  ケース {case}: {prob}")
        return

    # Selectorオブジェクトの作成
    selector = Selector(completed_choice_df, p_case)

    # 最適な質問を選択
    question_id = selector.select_best_question()
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
    recursive_question_selection(completed_choice_df, p_case)


recursive_question_selection(completed_choice_df, data["p_case"])
