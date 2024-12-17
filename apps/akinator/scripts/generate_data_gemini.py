import random
import json
import time
import concurrent.futures
import google.generativeai as genai
from infra.ai import llm
import threading
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
それぞれの選択肢の確率は少数点第一位まで表し、合計が1になるようにしてください。

Use this JSON schema:

質問: {question}
ケース: {case}

Recipe = {{'yes': float, 'probably_yes': float, 'dont_know': float, 'probably_no': float, 'no': float}}
Return: Recipe
"""


class Distribution(typing.TypedDict):
    yes: float
    probably_yes: float
    dont_know: float
    probably_no: float
    no: float


def validate_distribution(distribution: Distribution):
    total = 0
    for key, value in distribution.items():
        total += value  # type: ignore
    if not (0.99999 <= total <= 1.00001):
        raise ValueError(
            f"Probability distribution must sum to 1, got {total}")
    return distribution


genai.configure(api_key=llm.GOOGLE_CLOUD_API_KEY)
model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
generation_config = genai.GenerationConfig(
    response_mime_type="application/json",
    response_schema=Distribution,
    max_output_tokens=16  # 簡単な分布を出力するだけなので、トークン数を制限
)


def find_json(answer_string: str):
    start = answer_string.find("{")
    end = answer_string.rfind("}")
    return answer_string[start:end+1]


def ask_llm(case, question, max_retries=5, retry_delay=10):
    retries = 0
    answer_string = ""
    while retries < max_retries:
        prompt = prompt_template.format(case=case, question=question)
        try:
            response = model.generate_content(prompt)
            answer_string = response.text.strip()
            json_string = find_json(answer_string)

            distribution: Distribution = json.loads(json_string)
            validate_distribution(distribution)
            return distribution
        except Exception as e:
            retries += 1
            print(
                f"Error processing case: {case}, question: {question} (Attempt {retries}/{max_retries})")
            print(f"Response: {answer_string}")
            print(f"Error: {e}")
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    print(f"Failed after {max_retries} attempts.")
    exit(1)


# 出力ファイル名
output_filepath = "../out/gen.json"
# ファイルアクセス用のロック
file_lock = threading.Lock()


# JSONファイルにデータを追記する関数（スレッドセーフ）
def append_to_json(filepath, data):
    with file_lock:
        try:
            with open(filepath, "r+", encoding="utf-8") as f:
                try:
                    file_data = json.load(f)
                    if not isinstance(file_data, list):
                        file_data = []
                except json.JSONDecodeError:
                    file_data = []

                # 既存のデータから、data["ケース"]と同じ"ケース"の値を持つ辞書を探す
                existing_case_data = next(
                    (item for item in file_data if item["ケース"] == data["ケース"]), None)

                if existing_case_data:
                    # 既存のデータがあれば、"質問リスト"を更新
                    existing_case_data["質問リスト"].extend(data["質問リスト"])
                else:
                    # 既存のデータがなければ、新しいデータを追加
                    file_data.append(data)

                f.seek(0)  # ファイルの先頭に移動
                json.dump(file_data, f, ensure_ascii=False, indent=4)
                f.truncate()  # ファイルの残りの部分を削除
        except FileNotFoundError:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump([data], f, ensure_ascii=False, indent=4)


# スレッドプールを使用して同時に質問を処理する関数
def process_case(case):
    print(f"Processing case: {case}")
    # 質問の数をランダムに決定 (8~11個)
    num_questions = random.randint(15, 15)
    # ランダムに質問を選択
    selected_questions = random.sample(questions, num_questions)

    # スレッド数を調整
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_question = {executor.submit(
            ask_llm, case, question): question for question in selected_questions}
        for future in concurrent.futures.as_completed(future_to_question):
            question = future_to_question[future]
            try:
                distribution = future.result()
                qa_data = {"質問": question, "確率分布": distribution}
                print(
                    f"Case: {case}, Question: {question}, Distribution: {distribution}")
                # データをファイルに追記
                append_to_json(output_filepath, {
                               "ケース": case, "質問リスト": [qa_data]})

            except Exception as e:
                print(
                    f"Error processing question: {question} for case: {case}")
                print(f"Error: {e}")


# 各ケースに対して処理を行う
# ケースごとのスレッド数を調整
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_case, creatures)

print(f"JSONデータを出力しました: {output_filepath}")
