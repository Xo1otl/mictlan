import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 感情分析に特化した事前学習済みモデルをロード
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# モデルを評価モードに設定
model.eval()

# テキストのリスト
texts = [
    "I love programming",
    "I hate bugs",
    "Python is my favorite language",
    "This is the worst day ever"
]

# テキストをトークナイズしてテンソルに変換
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 推論実行
with torch.no_grad():
    outputs = model(**inputs)

# ログitsをソフトマックスでクラス確率に変換
probabilities = torch.softmax(outputs.logits, dim=-1)

# 結果を表示
predicted_classes = torch.argmax(probabilities, dim=1)
for text, prediction, prob in zip(texts, predicted_classes, probabilities):
    print(
        f"Text: {text} | Predicted class: {prediction.item()} | Probabilities: {prob.tolist()}")
