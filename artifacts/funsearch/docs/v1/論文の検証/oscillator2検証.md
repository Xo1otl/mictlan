# oscillator2

## どの部分に着目してどのような検証を行ったか

* **着目した点**: oscillator1とbactgrowを検証した後、念のため全部やっとこうと思って検証した
* **行った実装**: qwen2.5-coder を用いて探索を実装した
* **何が明確になったか**: 122回目で十分誤差が小さくなる関数を発見した

## 発見された関数
```python
def equation_v2(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
        omega0 = params[0]  # natural frequency
        gamma = params[1]   # damping ratio
        F0 = params[2]     # amplitude of driving force
        omega_d = params[3] # angular frequency of driving force
        alpha = params[4]   # nonlinear term coefficient for position
        beta = params[5]    # nonlinear term coefficient for velocity
        delta = params[6]   # additional damping term coefficient
        eta = params[7]     # coupling parameter between position and velocity
        phi = params[8]     # phase shift in driving force
        chi = params[9]     # additional nonlinear term coefficient
        dvdt = -omega0**2 * x - 2 * gamma * omega0 * v - delta * v + F0 * np.cos(omega_d * t + phi) + alpha * (x**2) + beta * (v**2) + eta * x * v + chi * (x**3)
        return dvdt
```

## 誤差表

### 今回
| Dataset | Loss Value                |
|---------|---------------------------|
| train   | -1.0750121293416077e-05   |
| id      | -1.0760972575998216e-05   |
| ood     | -0.042156957377099846     |

### 先行研究
#### oodとid
![alt text](oodとidの誤差表.png)
#### train
![alt text](探索中の誤差の遷移.png)

## 考察
oodに対してあまり精度がよくないものがみつかった、切り上げるの早すぎたかもしれない
