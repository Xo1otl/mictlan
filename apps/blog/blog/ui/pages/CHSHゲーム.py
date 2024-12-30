import math
import random
from typing import Tuple, NamedTuple
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def display_circle_with_specific_marks():
    def deg2rad(deg):
        return deg * np.pi / 180

    def circle_point(radius, angle_rad):
        x = radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)
        return x, y
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_aspect('equal')
    circle = plt.Circle((0, 0), 1, color='r', fill=False)  # type: ignore
    ax.add_patch(circle)
    angles_deg = [135/2, 135/2 + 90, 135/2 + 90 + 45, 135/2 + 90 + 45 + 90]
    angles_rad = [deg2rad(angle) for angle in angles_deg]
    for angle_rad in angles_rad:
        x, y = circle_point(1, angle_rad)
        ax.plot(x, y, 'bo')  # 青い丸で印を付ける
    ax.set_xlim([-1.2, 1.2])  # type: ignore
    ax.set_ylim([-1.2, 1.2])  # type: ignore
    ax.axis('off')
    st.pyplot(fig, use_container_width=False)


"""
# CHSHゲーム(トランプよりルール簡単)
"""

st.video("https://youtu.be/v7jctqKsUMA?si=aMuLx_V1fyQe6sVP")

r"""
以下のような`ルール`に従って、`量子もつれ`を使用した`戦略`で勝率を計算します。

## ルール

- 2人のプレイヤーと、一人のレフェリーが存在する
- それぞれのプレイヤーが、レフェリーから1か0を受け取る
- それぞれのプレイヤーは、レフェリーに対して1か0を返す(受け取った数字とは関係なく)
- レフェリーの渡した値が1, 1の場合、プレイヤーは違う値を返す必要がある
- それ以外の組み合わせに対して、プレイヤーは同じ値を返す必要がある

## 量子もつれ

- 観測に対して、必ず0,1のいずれかを出力する
- $sin^2(\frac \theta 2)$の確率で同じ値を返す
    - 同じ角度から観測すると必ず異なる値を返す
    - 正反対の角度から観測するとかならず同じ値を返す

## 戦略
"""

strategy, image = st.columns([2, 1])

with strategy:
    r"""
    - 両者が1, 1を受け取った場合のみ、観測する角度が$\frac \pi 4$になり、それ以外では$\frac {3\pi} 4$になるような設定を行う
        - 一周$2\pi$の観測範囲を持つ円形の観測機を使用
        - プレイヤー1が量子もつれを観測する角度を、受け取った値が0の時$\frac {3\pi} 8$、受け取った値が1の時$\frac {7\pi} 8$とする
        - プレイヤー2が量子もつれを観測する角度を、受け取った値が1の時$\frac {9\pi} 8$、受け取った値が0の時$\frac {13\pi} 8$とする
        - 1, 1の場合のみ間の角度が$\frac {2\pi} 8$で、それ以外の場合は$\frac {6\pi} 8$となる
    """

with image:
    display_circle_with_specific_marks()

r"""
- 観測によって量子もつれが出力した値をレフェリーに送信する
    - 1, 1を受け取った場合に、約15％の確率で同じ値を返し、約85％の確率で違う値を返すため、勝率は85％である
    $$
    \theta = \frac {2\pi} 8 \\
    sin^2(\frac \theta 2) = sin^2 (\frac \pi 8) = 0.146... = 0.15
    $$
    - 1, 1以外の組み合わせについて、約85％の確率で同じ値を返すため、勝率は85％である
    $$
    \theta = \frac {6\pi} 8 \\
    sin^2(\frac \theta 2) = sin^2 (\frac {3\pi} 8) = 0.8535... = 0.85
    $$
    - よって、すべての場合について、勝率は85％になる

## シミュレーション
以下のコードが実行されます！
```python
import math
import random
from typing import Tuple, NamedTuple

class QuantumMeasurement(NamedTuple):
    '''量子測定の結果を表現する型'''
    value: int  # 測定結果 (0 or 1)
    angle: float  # 測定角度（ラジアン）

class EntangledParticles:
    '''量子もつれをシミュレートするクラス'''

    def __init__(self):
        self.mesurement: QuantumMeasurement | None = None

    def measure(self, angle: float) -> int:
        '''
        特定の角度から量子もつれを観測する

        Args:
            angle (float): 観測する角度（ラジアン）

        Returns:
            int: 観測結果（0 or 1）
        '''
        if not self.mesurement:
            self.mesurement = QuantumMeasurement(
                angle=angle,
                value=random.randint(0, 1)
            )
            return self.mesurement.value
        else:
            angle_diff = abs(angle - self.mesurement.angle)
            same_prob = math.sin(angle_diff / 2) ** 2
            return self.mesurement.value if random.random() < same_prob else 1 - self.mesurement.value

class QuantumPlayer:
    '''量子戦略を使用するCHSHゲームのプレイヤー'''

    def __init__(self, angle_for_0: float, angle_for_1: float, particles: EntangledParticles):
        '''
        Args:
            angle_for_0 (float): 0を受け取った時の観測角度
            angle_for_1 (float): 1を受け取った時の観測角度
        '''
        self.angle_for_0 = angle_for_0
        self.angle_for_1 = angle_for_1
        self.particle = particles

    def answer(self, received_value: int) -> int:
        '''
        量子もつれを観測して応答を返す

        Args:
            quantum (QuantumEntanglement): 観測する量子もつれ
            received_value (int): レフェリーから受け取った値（0 or 1）

        Returns:
            int: 観測結果（0 or 1）
        '''
        angle = self.angle_for_0 if received_value == 0 else self.angle_for_1
        return self.particle.measure(angle)

class Referee:
    '''CHSHゲームの審判'''

    def play_round(self, player1: QuantumPlayer, player2: QuantumPlayer) -> Tuple[int, int, bool]:
        '''
        1ラウンドのゲームを実行する

        Args:
            player1 (Player): プレイヤー1
            player2 (Player): プレイヤー2

        Returns:
            Tuple[int, int, bool]: 送信した数値とゲームの勝敗
        '''
        # ランダムに入力を選択
        value1, value2 = random.randint(0, 1), random.randint(0, 1)
        response1 = player1.answer(value1)
        response2 = player2.answer(value2)

        # 1,1の場合は異なる値を返す必要がある
        if value1 == 1 and value2 == 1:
            return value1, value2, response1 != response2
        # それ以外は同じ値を返す必要がある
        return value1, value2, response1 == response2

def run_chsh(n_trials: int = 1000) -> Tuple[float, dict]:
    '''
    CHSHゲームの実験を実行する

    Args:
        n_trials (int): 試行回数

    Returns:
        Tuple[float, dict]: 勝率と各入力の組み合わせごとの勝率
    '''
    referee = Referee()

    # 結果を記録する辞書
    results = {
        (i, j): {'wins': 0, 'total': 0, 'win_rate': 0.0} for i in (0, 1) for j in (0, 1)
    }
    total_wins = 0

    for _ in range(n_trials):
        particles = EntangledParticles()
        player1 = QuantumPlayer(
            angle_for_0=3 * math.pi/8,
            angle_for_1=7*math.pi/8,
            particles=particles
        )
        player2 = QuantumPlayer(
            angle_for_0=13 * math.pi/8,
            angle_for_1=9*math.pi/8,
            particles=particles
        )
        value1, value2, win = referee.play_round(player1, player2)

        # 結果を記録
        results[(value1, value2)]['total'] += 1
        if win:
            results[(value1, value2)]['wins'] += 1
            total_wins += 1

    # 各入力の組み合わせごとの勝率を計算
    for key in results:
        if results[key]['total'] > 0:
            results[key]['win_rate'] = results[key]['wins'] / \
                results[key]['total']

    return total_wins / n_trials, results

total_win_rate, detailed_results = run_chsh(10000)

print(f'Total win rate: {total_win_rate:.1%}')
print('\nDetailed results:')
for (v1, v2), result in detailed_results.items():
    if result['total'] > 0:
        print(f'Input ({v1}, {v2}): {result["win_rate"]:.1%} '
              f'({result["wins"]}/{result["total"]} wins)')
```
"""


class QuantumMeasurement(NamedTuple):
    '''量子測定の結果を表現する型'''
    value: int  # 測定結果 (0 or 1)
    angle: float  # 測定角度（ラジアン）


class EntangledParticles:
    '''量子もつれをシミュレートするクラス'''

    def __init__(self):
        self.mesurement: QuantumMeasurement | None = None

    def measure(self, angle: float) -> int:
        '''
        特定の角度から量子もつれを観測する

        Args:
            angle (float): 観測する角度（ラジアン）

        Returns:
            int: 観測結果（0 or 1）
        '''
        if not self.mesurement:
            self.mesurement = QuantumMeasurement(
                angle=angle,
                value=random.randint(0, 1)
            )
            return self.mesurement.value
        else:
            angle_diff = abs(angle - self.mesurement.angle)
            same_prob = math.sin(angle_diff / 2) ** 2
            return self.mesurement.value if random.random() < same_prob else 1 - self.mesurement.value


class QuantumPlayer:
    '''量子戦略を使用するCHSHゲームのプレイヤー'''

    def __init__(self, angle_for_0: float, angle_for_1: float, particles: EntangledParticles):
        '''
        Args:
            angle_for_0 (float): 0を受け取った時の観測角度
            angle_for_1 (float): 1を受け取った時の観測角度
        '''
        self.angle_for_0 = angle_for_0
        self.angle_for_1 = angle_for_1
        self.particle = particles

    def answer(self, received_value: int) -> int:
        '''
        量子もつれを観測して応答を返す

        Args:
            quantum (QuantumEntanglement): 観測する量子もつれ
            received_value (int): レフェリーから受け取った値（0 or 1）

        Returns:
            int: 観測結果（0 or 1）
        '''
        angle = self.angle_for_0 if received_value == 0 else self.angle_for_1
        return self.particle.measure(angle)


class Referee:
    '''CHSHゲームの審判'''

    def play_round(self, player1: QuantumPlayer, player2: QuantumPlayer) -> Tuple[int, int, bool]:
        '''
        1ラウンドのゲームを実行する

        Args:
            player1 (Player): プレイヤー1
            player2 (Player): プレイヤー2

        Returns:
            Tuple[int, int, bool]: 送信した数値とゲームの勝敗
        '''
        # ランダムに入力を選択
        value1, value2 = random.randint(0, 1), random.randint(0, 1)
        response1 = player1.answer(value1)
        response2 = player2.answer(value2)

        # 1,1の場合は異なる値を返す必要がある
        if value1 == 1 and value2 == 1:
            return value1, value2, response1 != response2
        # それ以外は同じ値を返す必要がある
        return value1, value2, response1 == response2


def run_chsh(n_trials: int = 1000) -> Tuple[float, dict]:
    '''
    CHSHゲームの実験を実行する

    Args:
        n_trials (int): 試行回数

    Returns:
        Tuple[float, dict]: 勝率と各入力の組み合わせごとの勝率
    '''
    referee = Referee()

    # 結果を記録する辞書
    results = {
        (i, j): {'wins': 0, 'total': 0, 'win_rate': 0.0} for i in (0, 1) for j in (0, 1)
    }
    total_wins = 0

    for _ in range(n_trials):
        particles = EntangledParticles()
        player1 = QuantumPlayer(
            angle_for_0=3 * math.pi/8,
            angle_for_1=7*math.pi/8,
            particles=particles
        )
        player2 = QuantumPlayer(
            angle_for_0=13 * math.pi/8,
            angle_for_1=9*math.pi/8,
            particles=particles
        )
        value1, value2, win = referee.play_round(player1, player2)

        # 結果を記録
        results[(value1, value2)]['total'] += 1
        if win:
            results[(value1, value2)]['wins'] += 1
            total_wins += 1

    # 各入力の組み合わせごとの勝率を計算
    for key in results:
        if results[key]['total'] > 0:
            results[key]['win_rate'] = results[key]['wins'] / \
                results[key]['total']

    return total_wins / n_trials, results


total_win_rate, detailed_results = run_chsh(10000)

st.write(f'Total win rate: {total_win_rate:.1%}')
st.write('\nDetailed results:')
for (v1, v2), result in detailed_results.items():
    if result['total'] > 0:
        st.write(f'Input ({v1}, {v2}): {result["win_rate"]:.1%} '
                 f'({result["wins"]}/{result["total"]} wins)')
