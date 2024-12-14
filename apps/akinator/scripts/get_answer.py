import json
import argparse


def get_probabilities(data, creature):
    return {q: a[creature] for q, a in data.items() if creature in a}


parser = argparse.ArgumentParser(
    description="Extract probability distributions for a specific creature from a JSON dataset.")
parser.add_argument("creature", type=str, nargs='?', default="ハシビロコウ",
                    help="The creature to extract data for (default: ハシビロコウ)")
args = parser.parse_args()

try:
    with open("../out/dataset_transformed.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error: {e}")
    exit()

probabilities = get_probabilities(
    json_data["p_choice_given_case_question"], args.creature)

if not probabilities:
    print(f"No data found for creature: {args.creature}")
    exit()

for question, probs in probabilities.items():
    print(f"質問: {question}")
    print(f"{args.creature}の確率分布:")
    for response, probability in probs.items():
        print(f"  {response}: {probability}")
    print("-" * 20)
