import random
import json
from ollama import chat
from pydantic import BaseModel, Field, model_validator
from typing import Dict
import time
import concurrent.futures
import google.generativeai as genai
from infra.ai import llm

genai.configure(api_key=llm.GOOGLE_CLOUD_API_KEY)

questions = [
    "食べられるものですか？",
    "生き物ですか？",
    "主に人が使用するものですか？",
    "身体に作用するものですか？",
    "元々、自然にあったものですか？",
    "特定の季節やイベントに関連しますか？",
    "サイズは大きいですか？",
    "重いですか？(3kg以上)",
    "高価なものですか？(1万円以上)",
    "柔らかいものですか？",
    "表面に凹凸や模様がありますか？",
    "強い香りを放ちますか？",
    "主に赤やオレンジ系統の色ですか？",
    "主に青や緑系統の色ですか？",
    "丸みを帯びた形ですか？",
    "複数の部品で構成されますか？",
    "動きますか？",
    "音を発するものですか？",
    "光を発したり反射しますか？",
    "危険なもの、または注意が必要ですか？"
]

animals = [
    "犬", "猫", "ウサギ", "ハムスター", "鳥", "亀", "金魚", "ライオン", "虎", "象",
    "キリン", "パンダ", "コアラ", "サル", "熊", "鹿", "キツネ", "タヌキ", "イノシシ", "馬",
    "牛", "羊", "ヤギ", "豚", "ニワトリ", "アヒル", "フクロウ", "ワシ", "タカ", "ヘビ"
]

foods = [
    "リンゴ", "ミカン", "バナナ", "ブドウ", "イチゴ", "メロン", "スイカ", "桃", "梨", "柿",
    "オレンジ", "グレープフルーツ", "レモン", "キウイ", "マンゴー", "パイナップル", "アボカド", "サクランボ", "栗", "サツマイモ",
    "ジャガイモ", "玉ねぎ", "ニンジン", "トマト", "キュウリ", "キャベツ", "白菜", "ほうれん草", "鶏肉", "豚肉"
]

others = [
    "ティッシュペーパー", "トイレットペーパー", "洗剤", "シャンプー", "歯磨き粉", "ゴミ袋", "ラップ", "掃除機", "洗濯機", "冷蔵庫",
    "電子レンジ", "Tシャツ", "ジーンズ", "スニーカー", "ベッド", "ソファ", "テーブル", "椅子", "カーテン", "クッション",
    "時計", "テレビ", "スマートフォン", "パソコン", "カメラ", "ゲーム機", "ルーター", "プリンター", "コーヒーメーカー", "アイロン"
]

products = animals + foods + others

prompt_template = """
あなたは市場調査員です。与えられたケースと質問について、一般的な回答者がどのように答えるかを予測し、その回答の確率分布をyes, probably_yes, probably_no, noの4つの選択肢で示してください。

回答者像:
質問がケースと少しでも関連性がある場合、できるだけyes, probably_yes, probably_no, noのいずれかを選択します。
ケースから直接的には判断できない場合でも、常識的に考えて判断できる場合は、その知識に基づいて、yes, probably_yes, probably_no, noのいずれかを選択しようと努力します。
どうしても判断できない場合にのみ、dont_knowを選択します。

質問: {question}
ケース: {case}

出力形式 (JSON):

```json
{{
    "yes": float,
    "probably_yes": float,
    "dont_know": float,
    "probably_no": float,
    "no": float,
}}
```"""


class Distribution(BaseModel):
    yes: float = Field(ge=0, le=1)
    probably_yes: float = Field(ge=0, le=1)
    dont_know: float = Field(ge=0, le=1)
    probably_no: float = Field(ge=0, le=1)
    no: float = Field(ge=0, le=1)
    # reason: str = Field(default="")

    @model_validator(mode='after')
    def validate_total(cls, values):
        total = sum(value for value in values.model_dump().values()
                    if isinstance(value, (int, float)))
        if not (0.99999 <= total <= 1.00001):
            raise ValueError(
                f"Probability distribution must sum to 1, got {total}")
        return values


def ask_llm(case, question, model_name="gemini-1.5-pro-latest", max_retries=3, retry_delay=1):
    """
    ケースと質問に基づいて確率分布を生成する関数 (GoogleのGeminiを使用)

    Args:
      case: ケース (例: "正義")
      question: 質問 (例: "それは、多くの人にとって価値があると一般的に考えられていますか？")
      model_name: 使用するGeminiモデルの名前
      max_retries: 最大リトライ回数
      retry_delay: リトライ間隔（秒）

    Returns:
      確率分布を表す辞書 (例: {"yes": 0.7, "probably_yes": 0.2, "dont_know": 0.05, "probably_no": 0.03, "no": 0.02})
    """
    retries = 0
    model = genai.GenerativeModel(model_name)
    while retries < max_retries:
        prompt = prompt_template.format(case=case, question=question)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                # response_schema=Distribution
            ),
        )
        try:
            # response.textからJSON部分を抽出
            json_str = response.text.strip("`").strip("json").strip()
            distribution = Distribution.model_validate_json(
                json_str).model_dump()

            return distribution
        except Exception as e:
            retries += 1
            print(
                f"Error processing case: {case}, question: {question} (Attempt {retries}/{max_retries})")
            print(f"Response: {response.text}")
            print(f"Error: {e}")
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    print(f"Failed after {max_retries} attempts.")
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

# スレッドプールを使用して同時に質問を処理する関数


def process_case(case):
    print(f"Processing case: {case}")
    # 質問の数をランダムに決定 (8~11個)
    num_questions = random.randint(8, 11)
    # ランダムに質問を選択
    selected_questions = random.sample(questions, num_questions)

    # 質問と回答、確率分布を格納する辞書
    qa_list = []
    # スレッド数を調整
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_question = {executor.submit(
            ask_llm, case, question): question for question in selected_questions}
        for future in concurrent.futures.as_completed(future_to_question):
            question = future_to_question[future]
            try:
                distribution = future.result()
                qa_list.append({"質問": question, "確率分布": distribution})
                print(
                    f"Case: {case}, Question: {question}, Distribution: {distribution}")
            except Exception as e:
                print(
                    f"Error processing question: {question} for case: {case}")
                print(f"Error: {e}")

    return {
        "ケース": case,
        "質問リスト": qa_list
    }


# 各ケースに対して処理を行う
# ケースごとのスレッド数を調整
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    results = executor.map(process_case, products)

# 結果をJSONファイルに追記
for result in results:
    append_to_json(output_filepath, result)

print(f"JSONデータを出力しました: {output_filepath}")
