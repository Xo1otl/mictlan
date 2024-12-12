import random
import json
from ollama import chat
from pydantic import BaseModel, Field, model_validator
from typing import Dict
import time

questions = [
    "Is this product a food item?",
    "Can this product be stored at room temperature?",
    "Is this product primarily used by women?",
    "Is this product often purchased for less than 1,000 yen?",
    "Is this product a consumable?",
    "Is this product primarily used indoors?",
    "Does this product use electricity?",
    "Is this product likely to be used by children?",
    "Is this product often sold in packaging with Japanese text?",
    "Does this product often weigh more than 1 kg?",
    "Does the demand for this product increase in relation to a specific season?",
    "Does this product often have an expiration date or a best-by date after opening?",
    "Does this product require assembly or installation?",
    "Is it difficult to purchase or use this product without a specific qualification or license?",
    "Does this product sometimes contain ingredients that may cause allergies?",
    "Does using this product require special tools or equipment?",
    "Are parts of this product made from recyclable materials?",
    "Does this product require care or maintenance?",
    "Is this product often purchased through online shopping?",
    "Can this product be used relatively easily without specialized knowledge or skills?"
]

products = [
    "Rice", "Bread", "Noodles", "Meat", "Fish", "Vegetables", "Fruits", "Eggs", "Dairy products", "Tofu",
    "Natto", "Pickles", "Canned food", "Retort pouch food", "Frozen food", "Seasonings", "Snacks", "Beverages",
    "Alcohol", "Coffee", "Tea", "Japanese tea", "Soup", "Cereal", "Jam", "Honey",
    "Olive oil", "Chocolate", "Ice cream", "Yogurt", "Detergent", "Fabric softener", "Soap",
    "Shampoo", "Conditioner", "Toothpaste", "Toothbrush", "Toilet paper", "Tissues",
    "Paper towels", "Plastic wrap", "Aluminum foil", "Trash bags", "Batteries", "Light bulbs", "Flashlight",
    "Cleaning tools", "Laundry supplies", "Insecticide", "Air freshener", "T-shirt", "Shirt", "Blouse", "Sweater",
    "Cardigan", "Jacket", "Coat", "Pants", "Skirt", "Dress", "Underwear", "Socks",
    "Shoes", "Hat", "Muffler", "Gloves", "Belt", "Necktie", "Scarf", "Accessories",
    "Notebook", "Pen", "Pencil", "Eraser", "Ruler", "Scissors", "Glue", "Stapler",
    "Sticky notes", "File", "Book", "Magazine", "Newspaper", "CD", "DVD", "Game",
    "Toy", "Cosmetics", "Pet supplies", "Gardening supplies", "DIY supplies", "Sports equipment",
    "Outdoor equipment", "Travel goods", "Gift items", "Seasonal products", "Flowers", "Medicine", "Postage stamps", "Lottery tickets"
]

prompt_template = """
You are an expert in survey data analysis. Please predict the realistic "probability distribution" of answers when the following "question" is asked to a large number of Japanese people about the given "case".

Consider the common knowledge and general perception of ordinary Japanese people towards the {case} in Japan.

The sum of the probabilities for all five options must be 1 (100%). Each probability should be a float between 0.0 and 1.0.

Question: {question}
Case: {case}

Please output the probability distribution in JSON format as shown below:
{{
    "yes": float,
    "probably_yes": float,
    "dont_know": float,
    "probably_no": float,
    "no": float
}}
"""


class Distribution(BaseModel):
    yes: float = Field(ge=0, le=1)
    probably_yes: float = Field(ge=0, le=1)
    dont_know: float = Field(ge=0, le=1)
    probably_no: float = Field(ge=0, le=1)
    no: float = Field(ge=0, le=1)

    @model_validator(mode='after')
    def validate_total(cls, values):
        total = sum(values.model_dump().values())
        if not (0.99999 <= total <= 1.00001):
            raise ValueError(
                f"Probability distribution must sum to 1, got {total}")
        return values


def ask_llm(case, question, model_name="qwq", max_retries=3, retry_delay=1):
    """
    ケースと質問に基づいて確率分布を生成する関数 (Ollamaを使用)

    Args:
      case: ケース (例: "正義")
      question: 質問 (例: "それは、多くの人にとって価値があると一般的に考えられていますか？")
      model_name: 使用するOllamaモデルの名前
      max_retries: 最大リトライ回数
      retry_delay: リトライ間隔（秒）

    Returns:
      確率分布を表す辞書 (例: {"yes": 0.7, "probably_yes": 0.2, "dont_know": 0.05, "probably_no": 0.03, "no": 0.02})
    """
    retries = 0
    while retries < max_retries:
        prompt = prompt_template.format(case=case, question=question)
        response = chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            format=Distribution.model_json_schema()  # type: ignore
        )  # type: ignore
        try:
            distribution = Distribution.model_validate_json(
                response['message']['content']).model_dump()
            return distribution
        except Exception as e:
            retries += 1
            print(
                f"Error processing case: {case}, question: {question} (Attempt {retries}/{max_retries})")
            print(f"Response: {response['message']['content']}")
            print(f"Error: {e}")
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    print(f"Failed after {max_retries} attempts.")
    # リトライ失敗時は均等な分布を返す
    exit(1)


# JSONファイルにデータを追記する関数
def append_to_json(filepath, data):
    try:
        with open(filepath, "r+", encoding="utf-8") as f:
            try:
                file_data = json.load(f)
                if not isinstance(file_data, list):
                    file_data = []
            except json.JSONDecodeError:
                file_data = []
            file_data.append(data)
            f.seek(0)  # ファイルの先頭に移動
            json.dump(file_data, f, ensure_ascii=False, indent=4)
            f.truncate()  # ファイルの残りの部分を削除
    except FileNotFoundError:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([data], f, ensure_ascii=False, indent=4)


# 出力ファイル名
output_filepath = "output.json"

# 各ケースに対して処理を行う
for i, case in enumerate(products):
    print(f"Processing case {i+1}/{len(products)}: {case}")
    # 質問の数をランダムに決定 (5~10個)
    num_questions = random.randint(8, 11)
    # ランダムに質問を選択
    selected_questions = random.sample(questions, num_questions)

    # 質問と回答、確率分布を格納する辞書
    qa_list = []
    for question in selected_questions:
        # 確率分布を取得 (Ollamaを使用)
        distribution = ask_llm(case, question)
        qa_list.append({"質問": question, "確率分布": distribution})
        print(f"Question: {question}, Distribution: {distribution}")

    # JSONファイルに追記
    append_to_json(output_filepath, {
        "ケース": case,
        "質問リスト": qa_list
    })

print(f"JSONデータを出力しました: {output_filepath}")
