import numpy as np

def calculate_entropy(probabilities):
  """エントロピーを計算する関数"""
  entropy = 0
  for p in probabilities:
    if p > 0:
      entropy -= p * np.log2(p)
  return entropy

def update_probability(p_apple, p_watermelon, p_heavy_given_apple, p_heavy_given_watermelon, answer):
    """ベイズ推定で確率を更新する関数"""
    # りんご、スイカの事前確率
    prior = np.array([p_apple, p_watermelon])

    # 尤度
    if answer == "yes":
        likelihood = np.array([p_heavy_given_apple, p_heavy_given_watermelon])
    else:
        likelihood = np.array([1 - p_heavy_given_apple, 1 - p_heavy_given_watermelon])

    # りんご、スイカそれぞれの周辺尤度
    evidence = np.sum(prior * likelihood)

    # 事後確率
    posterior = (prior * likelihood) / evidence

    return posterior[0], posterior[1]

# 初期確率
p_apple = 0.5
p_watermelon = 0.5

# 最初の「重い」と答える確率
p_heavy_given_apple_initial = 0.5
p_heavy_given_watermelon_initial = 0.75

# 回答がわかった後の「重い」と答える確率
p_heavy_given_apple_known = 1.0
p_heavy_given_watermelon_known = 1.0

# エントロピーの履歴
entropy_history = []

# 質問を3回繰り返す
for i in range(3):
    # 現在のエントロピーを計算
    entropy = calculate_entropy([p_apple, p_watermelon])
    entropy_history.append(entropy)

    print(f"質問 {i+1} 回目:")
    print(f"  りんごの確率: {p_apple:.4f}")
    print(f"  スイカの確率: {p_watermelon:.4f}")
    print(f"  エントロピー: {entropy:.4f}")

    # 質問に対する回答をシミュレート（ここでは、スイカが存在すると仮定して「はい」と答える）
    # 実際にはりんごかスイカが選ばれて、その確率に従って「はい」か「いいえ」を答えるように書き換える必要がある
    #answer = "yes" if np.random.rand() < p_heavy_given_watermelon_initial else "no"
    # 修正：りんごとスイカの確率に基づいて、どちらのオブジェクトが存在するかを選択し、そのオブジェクトの「重い」と答える確率に基づいて回答を生成する
    if np.random.rand() < p_apple:
        # りんごが選ばれた場合
        if i == 0:
          answer = "yes" if np.random.rand() < p_heavy_given_apple_initial else "no"
        else:
          answer = "yes" if np.random.rand() < p_heavy_given_apple_known else "no"
    else:
        # スイカが選ばれた場合
        if i == 0:
          answer = "yes" if np.random.rand() < p_heavy_given_watermelon_initial else "no"
        else:
          answer = "yes" if np.random.rand() < p_heavy_given_watermelon_known else "no"

    print(f"  回答: {answer}")

    # 確率を更新
    if i == 0:
        p_apple, p_watermelon = update_probability(
            p_apple, p_watermelon, p_heavy_given_apple_initial, p_heavy_given_watermelon_initial, answer
        )
    else:
        p_apple, p_watermelon = update_probability(
            p_apple, p_watermelon, p_heavy_given_apple_known, p_heavy_given_watermelon_known, answer
        )

print("\nエントロピーの履歴:")
for i, entropy in enumerate(entropy_history):
    print(f"  {i+1} 回目の質問後: {entropy:.4f}")
