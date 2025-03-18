from syuron import dataset
from .model import ModelState, UseState, TrainStep, LossFn, Loss
from typing import List, NamedTuple, Tuple
from bayes_opt import BayesianOptimization
from tqdm import tqdm


def bayesian_optim(dataset: dataset.Dataset, use_state: UseState,
                   train_step: TrainStep, loss_fn: LossFn,
                   init_points: int = 5, n_iter: int = 25) -> Tuple[ModelState, Loss]:
    # train_and_evalをラップした関数。hidden_sizesの各要素は整数化して扱う。
    def evaluate_model(learning_rate: float, hidden_size_1: float, hidden_size_2: float, epochs: int) -> Tuple[ModelState, Loss]:
        op_params = OptimizableParams(
            learning_rate=learning_rate,
            hidden_sizes=[int(hidden_size_1), int(hidden_size_2)]
        )
        return train_and_eval(dataset, use_state, train_step, loss_fn, op_params, epochs)

    def objective(learning_rate: float, hidden_size_1: float, hidden_size_2: float) -> float:
        epochs = 1  # パラメータチューニング中は1エポックで評価する
        _, loss = evaluate_model(
            learning_rate, hidden_size_1, hidden_size_2, epochs)
        print(
            f"Validation loss (lr={learning_rate}, hidden1={int(hidden_size_1)}, hidden2={int(hidden_size_2)}): {loss}")
        # 損失を最小化するため、BayesianOptimizationでは符号反転して返す
        return -loss

    # 探索範囲（ハードコード）
    pbounds = {
        'learning_rate': (1e-6, 1e-1),
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
        best_params['learning_rate'],
        best_params['hidden_size_1'],
        best_params['hidden_size_2'],
        epochs=5
    )
    print("Final evaluation with best parameters (5 epochs):")
    print("Final loss:", final_loss)
    return final_state, final_loss


class OptimizableParams(NamedTuple):
    learning_rate: float
    hidden_sizes: List[int]


def train_and_eval(ds: dataset.Dataset, use_state: UseState, train_step: TrainStep, loss_fn: LossFn, op_params: OptimizableParams, epochs: int) -> Tuple[ModelState, Loss]:
    learning_rate, hidden_sizes = op_params

    # サンプルバッチから入力次元と出力次元を取得
    sample_batch = next(iter(ds))
    sample_inputs, sample_outputs = sample_batch
    input_size = sample_inputs.shape[-1]  # (batch_size, input_size)
    output_size = sample_outputs.shape[-1]  # (batch_size, output_size)

    # モデル初期化
    state = use_state(learning_rate, input_size, hidden_sizes, output_size)

    # 初期損失の計算
    batch = dataset.Batch(inputs=sample_inputs, outputs=sample_outputs)
    init_loss = loss_fn(state.params, batch, state.apply_fn)
    print("Initial loss:", init_loss)

    avg_loss = 0.0
    # 学習ループ
    for epoch in range(epochs):
        epoch_loss = 0.0
        count = 0
        for batch_data in tqdm(ds, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, outputs = batch_data
            batch = dataset.Batch(inputs=inputs, outputs=outputs)
            state, loss = train_step(state, batch)
            epoch_loss += loss
            count += 1
        avg_loss = epoch_loss / count
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss}")
    return state, avg_loss
