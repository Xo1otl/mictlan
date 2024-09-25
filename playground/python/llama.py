from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch

model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# 4bit量子化設定を作成
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4bit量子化を有効化
    bnb_4bit_compute_dtype=torch.float16  # 計算用データ型をfloat16に設定
)

# トークナイザーをロード
tokenizer = AutoTokenizer.from_pretrained(model_id)

# モデルを4bit量子化でロード
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,  # 4bit量子化設定を渡す
    device_map='auto',
    torch_dtype=torch.float16,  # 推論のデータ型をfloat16に設定
)

print(model.hf_device_map)

# パイプラインを作成
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

# 入力テキストを準備
input_text = (
    "You are a pirate chatbot who always responds in pirate speak!\n"
    "User: Who are you?\n"
    "Assistant:"
)

# テキストを生成
outputs = text_gen_pipeline(
    input_text,
    max_new_tokens=256,
    do_sample=True,
)

print(outputs[0]['generated_text'])  # type: ignore
