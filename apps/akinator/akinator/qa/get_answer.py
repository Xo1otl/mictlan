import json


def get_octopus_probabilities(data, creature):
    octopus_data = {}
    for question, answers in data.items():
        if creature in answers:
            octopus_data[question] = answers[creature]
    return octopus_data


json_data = json.loads(open("output_transformed.json",
                       "r", encoding="utf-8").read())

creature = "ハシビロコウ"

octopus_probabilities = get_octopus_probabilities(
    json_data["p_choice_given_case_question"], creature)

for question, probabilities in octopus_probabilities.items():
    print(f"質問: {question}")
    print(f"{creature}の確率分布:")
    for response, probability in probabilities.items():
        print(f"  {response}: {probability}")
    print("-" * 20)
