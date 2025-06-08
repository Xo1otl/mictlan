import random
import json
from ollama import chat
from pydantic import BaseModel, Field, model_validator
from typing import Dict
import time
import concurrent.futures
import google.genai
import typing_extensions as typing

questions = [
    "水の中で生活しますか？",
    "主に歩いて移動しますか？",
    "体長は1メートル以上ありますか？",
    "体に硬い殻はありますか？",
    "毛深いですか？",
    "肉食ですか？",
    "卵を産みますか？",
    "群れで生活しますか？",
    "色は鮮やかですか？",
    "夜間に活動する場合も多いですか？",
    "鳴き声で仲間とコミュニケーションをとりますか？",
    "身を守る時に、敵から見つかりにくくする工夫をしますか？",
    "体温調節機能はありますか？",
    "子供と大人で姿が大きく変わりますか？",
    "10年以上生きますか？",
    "冬眠しますか？",
    "実在する生き物ですか？",
    "絶滅していますか？",
    "光を発することができますか？",
    "季節によって長距離を移動しますか？",
    "特定の地域にしか生息していませんか？",
    "絶滅の危機に瀕していますか？",
    "道具を使うなど、知能が高いですか？",
    "人間と友好的な関係を築くことがありますか？",
    "他の生物に擬態しますか？",
    "足は5本以上ありますか？",
    "しっぽがありますか？",
    "肺呼吸ですか？",
    "毒を持っていますか？",
    "飛ぶことができますか？"
]

creatures = ["イヌ", "ゾウ", "チンパンジー", "カンガルー", "イルカ", "ラッコ", "サイ", "ユキヒョウ", "アルパカ", "ウマ", "ハト", "カラス", "ワシ", "ペンギン", "ダチョウ", "ヘビ", "ワニ", "カメ", "カエル", "サケ", "マグロ", "タコ", "チョウ", "カブトムシ", "トンボ", "アリ", "ホタル", "クモ", "ムカデ", "ミミズ", "エビ", "スズメ", "インコ", "オタマジャクシ", "キンギョ", "クワガタムシ", "ヒョウモントカゲモドキ", "ウーパールーパー", "イソギンチャク", "クラゲ", "ヒトデ", "ウニ", "シャコ", "ザリガニ", "アミメキリン", "エトピリカ", "カモノハシ", "クリオネ", "クジラ", "クマムシ", "コウモリ", "サンゴ", "チョウチンアンコウ", "ナマケモノ",
             "ハエトリグモ", "ハシビロコウ", "ハダカデバネズミ", "ヒト", "ミジンコ", "モグラ", "ティラノサウルス", "ブラキオサウルス", "プテラノドン", "マツ", "イチョウ", "タンポポ", "ヒマワリ", "アサガオ", "ユリ", "シイタケ", "マツタケ", "ドラゴン", "ユニコーン", "ペガサス", "フェニックス", "ウミウシ", "オオカミ", "カイコ", "ゴキブリ", "ダンゴムシ", "セミ", "トリケラトプス", "テントウムシ", "バッタ", "ヤドカリ", "カニ", "スギ", "サクラ", "チューリップ", "エノキタケ", "ケルベロス", "カマキリ", "カピバラ", "カタツムリ", "アザラシ", "イモリ", "カバ", "キリン", "コイ", "ツバメ", "ニワトリ", "ハチ", "パンダ", "ブタ", "ラクダ"]

prompt_template = """
以下の質問とケースについて大勢の人間にアンケートをとった時に、`yes`, `probably_yes`, `dont_know`, `probably_no`, `no`のそれぞれが選択される確率分布を予想してください。
すべての選択肢の合計は1になるようにしてください。

Use this JSON schema:

質問: {question}
ケース: {case}

Recipe = {{'yes': float, 'probably_yes': float, 'dont_know': float, 'probably_no': float, 'no': float}}
Return: Recipe
"""


class Distribution(BaseModel):
    yes: float = Field(ge=0, le=1)
    probably_yes: float = Field(ge=0, le=1)
    dont_know: float = Field(ge=0, le=1)
    probably_no: float = Field(ge=0, le=1)
    no: float = Field(ge=0, le=1)
    # reason: str = Field(default="")

    @model_validator(mode='after')
    def validate_total(cls, values):
        total = 0
        for key, value in values.model_dump().items():
            # if key != "dont_know" and not (0 < value <= 1):
            #     raise ValueError(
            #         f"Probability values must be grater 0, got {value}")
            # if key == "dont_know" and not (0.5 > value):
            #     raise ValueError(
            #         f"Probability of 'dont_know' must be less than 0.5, got {value}")
            total += value
        if not (0.99999 <= total <= 1.00001):
            raise ValueError(
                f"Probability distribution must sum to 1, got {total}")
        return values


def ask_llm(case, question, model_name="jaahas/gemma-2-9b-it-abliterated:latest", max_retries=3, retry_delay=1):
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
output_filepath = "../out/gen_ollama.json"


# スレッドプールを使用して同時に質問を処理する関数
def process_case(case):
    print(f"Processing case: {case}")
    # 質問の数をランダムに決定 (8~11個)
    num_questions = random.randint(15, 15)
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
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(process_case, creatures)

# 結果をJSONファイルに追記
for result in results:
    append_to_json(output_filepath, result)

print(f"JSONデータを出力しました: {output_filepath}")
