import json
from collections import defaultdict


def transform_data(data):
    """
    データを変換する関数。
    各ケースの出現確率を 1/全ケース数 で計算し、確率値をそのまま格納する。
    """
    p_case = {}
    p_choice_given_case_question = defaultdict(lambda: defaultdict(lambda: {}))

    # 全ケース数を計算
    num_cases = len(data)

    for item in data:
        case_name = item["ケース"]
        # 各ケースの出現確率を計算
        p_case[case_name] = 1 / num_cases

        for q_data in item["質問リスト"]:
            question = q_data["質問"]
            for choice, prob in q_data["確率分布"].items():
                # 確率をそのまま辞書に格納
                p_choice_given_case_question[question][case_name][choice] =\
                    prob + 0.05 / (1 + 0.05 * 5)

    return {
        "p_case": p_case,
        # defaultdictを通常の辞書に変換して出力
        "p_choice_given_case_question": {
            q: {c: dict(v) for c, v in cases.items()} for q, cases in p_choice_given_case_question.items()
        }
    }


# データの読み込み
with open("output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# データの変換
transformed_data = transform_data(data)

# データの出力
with open("output_transformed.json", "w", encoding="utf-8") as f:
    json.dump(transformed_data, f, indent=2, ensure_ascii=False)

print("データ変換が完了しました。")
