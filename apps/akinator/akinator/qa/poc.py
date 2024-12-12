import pandas as pd
import numpy as np

# --- データ作成 (is_it_animal の banana と watermelon の行を削除したデータ) ---
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
df_filled = df.copy()
df_filled["p_choice_given_case_question"] = df_filled.groupby(["question_id", "choice"], group_keys=False)[
    "p_choice_given_case_question"].apply(lambda group: group.fillna(weighted_mean(group, df_p_case)))

# --- エントロピーの計算 ---
# マルチインデックスを設定
df_filled = df_filled.set_index(["question_id", "case_id", "choice"])

# p_caseのDataFrameと結合
df_joined = df_filled.join(df_p_case.set_index("case_id"), on="case_id")

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

print(df_joined)


def calculate_entropy(probabilities):
    return -np.sum(np.where(probabilities > 0, probabilities * np.log2(probabilities), 0))


# 質問と選択肢でグループ化し、各グループ(場合)のエントロピーを計算
entropy_df = df_joined.groupby(["question_id", "choice"])[
    "p_case_given_choice_question"
].apply(calculate_entropy).reset_index()

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

entropy_df.rename(columns={"weighted_entropy": "entropy"}, inplace=True)
print(entropy_df)

# --- 最もエントロピーが小さい質問を選択 ---
best_question = entropy_df.loc[entropy_df["entropy"].idxmin()]
print(f"最もエントロピーが小さい質問: {best_question['question_id']}")

# --- 選択した質問に対する回答を取得 ---
# 例として、'yes' という回答が得られたと仮定します。
# 実際にはユーザーからの入力を受け取るか、シミュレーションによって回答を生成します。
answer = "yes"  # ここでは、例として "yes" を設定
print(f"選ばれた回答: {answer}")

# --- 事後確率の計算 ---
# 質問と回答に基づいて、事後確率を計算
posterior_probs = df_joined.loc[(
    best_question["question_id"], slice(None), answer), "p_case_given_choice_question"]

# インデックスをリセットして、case_id を列に戻す
posterior_probs = posterior_probs.reset_index(level="case_id")

# case_id ごとの確率を取得
posterior_probs = posterior_probs.set_index("case_id")[
    "p_case_given_choice_question"]

# 事後確率が 0 の場合、非常に小さい非ゼロの値に置き換える
posterior_probs = posterior_probs.where(
    posterior_probs != 0, 1e-10)

# 事後確率を正規化して、合計が 1 になるようにする
posterior_probs = posterior_probs / posterior_probs.sum()

print("新しい事後確率分布:")
print(posterior_probs)
