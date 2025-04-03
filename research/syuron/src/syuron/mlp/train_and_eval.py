from syuron import dataset
from .model import ModelState, UseState, TrainStep, LossFn, Loss
from typing import List, NamedTuple, Tuple
from tqdm import tqdm


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
            state, loss = train_step(state, batch, loss_fn)
            epoch_loss += loss
            count += 1
        avg_loss = epoch_loss / count
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss}")
    return state, avg_loss
