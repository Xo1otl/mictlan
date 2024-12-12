import json
from collections import defaultdict


def transform_data(data):
    p_case = {item["ケース"]: 1/len(data) for item in data}
    p_choice_given_case_question = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float)))

    for item in data:
        for q_data in item["質問リスト"]:
            for choice, prob in q_data["確率分布"].items():
                p_choice_given_case_question[q_data["質問"]
                                             ][item["ケース"]][choice] += prob / len(data)

    return {
        "p_case": p_case,
        "p_choice_given_case_question": {q: {c: dict(v) for c, v in cases.items()} for q, cases in p_choice_given_case_question.items()}
    }


# データの読み込みと変換
with open("output.json", "r", encoding="utf-8") as f:
    transformed_data = transform_data(json.load(f))

# データの出力
with open("output_transformed.json", "w", encoding="utf-8") as f:
    json.dump(transformed_data, f, indent=2, ensure_ascii=False)

print("データ変換が完了しました。")
