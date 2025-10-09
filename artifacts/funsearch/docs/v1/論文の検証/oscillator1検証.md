# oscillator1

## どの部分に着目してどのような検証を行ったか

* **着目した点**: oscillator1は比較的シンプルな例であり、最初の検証に向いている点に着目した
* **行った実装**: gemini3:12b を用いて探索を実装した
* **何が明確になったか**: 236回目で十分誤差が小さくなる関数を発見した

## 発見された関数
```python
def found_equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    acceleration = params[0] + params[1] * v - params[2] * \
        x + params[3] * x * v - params[4] * x**2 * np.sin(x)
    return acceleration
```

## 誤差表

### 今回
| Dataset | Loss Value                |
|---------|---------------------------|
| train   | -2.2259064280660823e-06   |
| id      | -2.2255817384575494e-06   |
| ood     | -1.608407728781458e-05    |

### 先行研究
#### oodとid
![alt text](oodとidの誤差表.png)
#### train
![alt text](探索中の誤差の遷移.png)

## 考察
oodに対して精度が高い
