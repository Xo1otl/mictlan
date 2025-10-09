# bactgrow v3

ミスってプログラムを停止するのを忘れ、丸一日geminiでマルチスレッドの探索をした結果、めちゃくちゃ高精度な式が発見された

# 23591回の探索で以下の誤差
-8.283971658023059e-10

# bactgrow

## どの部分に着目してどのような検証を行ったか

* **着目した点**: プログラムを停止し忘れた
* **行った実装**: gemini flashを一日中マルチスレッドに探索して、3万回ぐらい変異する実装
* **何が明確になったか**: 23591回目で、超高精度の関数が発見された

## 発見された関数
```python
def equation_v2(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    dv = params[0] * np.sin(params[1] * t) - params[2] * x - params[3] * v + params[4] * x**2 - params[5] * v**3 + params[6] * np.cos(params[7] * t) * v - params[8] * x * v + params[9] *
 np.exp(-params[3] * (x**2 + v**2)) * np.sin(t) + params[1] * np.tanh(x) # Included tanh(x)
    return dv
```

## 誤差表

### 今回
| データセット | 誤差 (loss)            |
|--------------|------------------------|
| train        | -8.283971658023059e-10  |
| id           | -2.429856216878641e-09  |
| ood          | -3.9922376613604577e-07 |

### 先行研究
#### oodとid
![alt text](oodとidの誤差表.png)
#### train
![alt text](探索中の誤差の遷移.png)

## 考察
特にバグることなく探索し続けられることがわかった

論文で発見されている関数より精度がたかい

これはちゃんと他のデータセットでも係数が見つかる形をしている

515円かかってしまった、意外と安い

bactgrowの方でミスってた方が嬉しかった、、、これもとから精度いいからあんま意味ない
