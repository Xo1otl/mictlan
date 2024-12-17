import json


def convert_json(input_file, output_file, category):
    """
    元の形式のJSONファイルを読み込み、新しい形式に変換して出力する関数。

    Args:
        input_file: 元の形式のJSONファイルパス
        output_file: 新しい形式のJSONファイルパス
        category: 分類する分野名 (例: "生き物")
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"エラー: {input_file} が見つかりません。")
        return
    except json.JSONDecodeError:
        print(f"エラー: {input_file} は有効なJSONファイルではありません。")
        return

    new_data = {
        category: data
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"変換完了: {output_file} に新しい形式のJSONを保存しました。")


# 使用例
input_file = '../out/dataset_transformed.json'  # 元のJSONファイル名
output_file = '../out/dataset.json'  # 新しいJSONファイル名
category = '生き物'  # 分野名

convert_json(input_file, output_file, category)
