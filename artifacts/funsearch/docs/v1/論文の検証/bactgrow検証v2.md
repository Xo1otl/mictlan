# bactgrow

## どの部分に着目してどのような検証を行ったか

* **着目した点**: gemini flash安いし速いのでやり直してみた
* **行った実装**: geminiを使う実装した
* **何が明確になったか**: 486回で見つけた、idとoodのデータで係数決められない問題発生したのでオプティマイザが全然万能じゃないことがわかった

## 発見された関数
```python
def equation_v2(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    mu_max = params[0]
    Ks = params[1]
    T_opt = params[2]
    pH_opt_low = params[3]
    pH_opt_high = params[4]
    Ea = params[5]  # Activation energy
    R = 8.314  # Gas constant
    # Monod equation for substrate limitation
    mu_s = (s / (Ks + s))
    # Temperature dependence (Arrhenius equation with optimum and deactivation)
    mu_temp = np.exp((Ea / R) * ((1 / T_opt) - (1 / temp))) * (1 / (1 + np.exp(
        params[6] * (T_opt - temp))))  # params[6] tunes deactivation steepness
    # pH dependence (more accurate dual range, optimum and range)
    mu_pH_low = 1 / (1 + (pH_opt_low / pH) ** params[7])
    mu_pH_high = 1 / (1 + (pH / pH_opt_high) ** params[8])
    # Combine pH effects. params[7,8] are pH exponents for lower and upper tolerance respectively.
    mu_pH = mu_pH_low * mu_pH_high
    # Inhibition term for high substrate concentrations (optional)
    Ki = params[9]
    # Avoid division by zero. If Ki is zero, no inhibition.
    inhibition = Ki / (Ki + s) if Ki > 0 else 1.0
    # Combine all effects (No maintenance or death rate) # Simplified model
    growth_rate = mu_max * mu_s * mu_temp * mu_pH * b * inhibition
    return growth_rate
```

## 目的関数を最小化する係数 (train.csv以外のデータを用いてoptimizeするとlossがnanになるエラーが発生する)
```
[ 4.79695157e+03  9.97774840e-01  3.09969435e+01  7.10961033e+00
  1.83658124e+01  8.80574701e+03 -9.35937532e-01 -2.23846922e+01
 -9.24085707e+00  2.87205527e+03]
```

## 誤差表

### 今回
| データセット | 誤差 (loss)            |
|--------------|------------------------|
| train        | -0.00031873956322669983 |
| id           | -0.00026723433450878556 |
| ood          | -0.0013777414169110084  |

### 先行研究
#### oodとid
![alt text](oodとidの誤差表.png)
#### train
![alt text](探索中の誤差の遷移.png)

## 考察
論文で発見されている関数より精度がたかい

なぜかtrain.csvでしか係数が見つからない、こういうことがよくあるのでオプティマイザの改良が必要だと思われる
