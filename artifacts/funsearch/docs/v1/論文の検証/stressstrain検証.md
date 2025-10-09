# stressstrain

## どの部分に着目してどのような検証を行ったか

* **着目した点**: oscillator1とbactgrowを検証した後、念のため全部やっとこうと思って検証した
* **行った実装**: qwen2.5-coder を用いて探索を実装した
* **何が明確になったか**: 37回目で先行研究より精度の良い関数を発見した

## 発見された関数
```python
def equation_v2(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    stress = (params[0] * strain + params[1] * temp + params[2] * strain**2 + params[3] * temp**2 + params[4] * strain * temp + params[5] * strain**3 + params[6] * temp**3 + params[7] * strain**2 * temp + params[8] * strain * temp**2) / (1 + params[9] * strain)
    return stress
```

## 誤差表

### 今回
| Dataset | Loss Value                |
|---------|---------------------------|
| train   | -0.002254487877904934     |
| id      | -0.0024281077105815633    |
| ood     | -0.001767546759416538     |

### 先行研究
#### oodとid
![alt text](oodとidの誤差表.png)
#### train
![alt text](探索中の誤差の遷移.png)

## 考察
oodに対しての方が精度が良い、qwenは単純な多項式得意なのかもしれない
