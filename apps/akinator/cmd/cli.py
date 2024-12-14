#!/usr/bin/env python

import torch
import json
from akinator import qa

# GPUが利用可能か確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

validation_dataset: qa.Dataset = {
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

dataset = json.loads(
    open("/workspaces/mictlan/apps/akinator/out/dataset_transformed.json").read())
context = qa.Context(dataset, device)
# context = qa.Context(validation_dataset, device)
context.complete()  # completeメソッドを呼び出して補完を実行

print("テンソルの形状:", context.tensor.shape)
print("テンソルの次元数:", context.tensor.ndim)
print("テンソルの要素数:", context.tensor.numel())
# print("選択肢分布", context.tensor[:, :, :,
#       context.probability_attr_to_idx["p_choice_given_case_question"]])

selector = qa.Selector(context)

qa.interactive_ask(context)
