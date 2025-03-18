from syuron import dataset
from bayes_opt import BayesianOptimization
from syuron import mlp
from typing import Tuple
from .train_and_eval import *


def bayesian_optim_and_eval(dataset: dataset.Dataset, init_points: int = 5, n_iter: int = 25) -> Tuple[mlp.ModelState, mlp.Loss]:
    # train_and_evalをラップした関数。hidden_sizesの各要素は整数化して扱う。
    def evaluate_model(learning_rate: float, hidden_size_1: float, hidden_size_2: float, epochs: int) -> Tuple[mlp.ModelState, mlp.Loss]:
        op_params = mlp.OptimizableParams(
            learning_rate=learning_rate,
            hidden_sizes=[int(hidden_size_1), int(hidden_size_2)]
        )
        return train_and_eval(dataset, epochs, op_params)

    def objective(log_learning_rate: float, hidden_size_1: float, hidden_size_2: float) -> float:
        epochs = 2  # パラメータチューニング中は2エポックで評価する
        learning_rate = 10 ** log_learning_rate
        _, loss = evaluate_model(
            learning_rate, hidden_size_1, hidden_size_2, epochs)
        print(
            f"Validation loss (lr={learning_rate}, hidden1={int(hidden_size_1)}, hidden2={int(hidden_size_2)}): {loss}")
        # 損失を最小化するため、BayesianOptimizationでは符号反転して返す
        return -loss

    # 探索範囲（ハードコード）
    pbounds = {
        'log_learning_rate': (-6, -1),
        'hidden_size_1': (32, 512),
        'hidden_size_2': (32, 512),
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        verbose=2,         # 各評価の詳細を出力
        random_state=42,   # 再現性のための乱数シード
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    print("Best parameters found:", optimizer.max)

    # 最適な結果が得られているかチェックする
    if optimizer.max is None:
        raise ValueError("Optimization did not yield a maximum result.")

    # 最終評価: 最適なハイパーパラメータで、エポック数5で学習し評価する
    best_params = optimizer.max['params']
    final_state, final_loss = evaluate_model(
        best_params['log_learning_rate'],
        best_params['hidden_size_1'],
        best_params['hidden_size_2'],
        epochs=10
    )
    print("Final evaluation with best parameters (5 epochs):")
    print("Final loss:", final_loss)
    return final_state, final_loss
