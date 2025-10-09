from funsearch import function
from funsearch import llmsr
from dataclasses import dataclass
from scipy.optimize import minimize
import numpy as np
from typing import NamedTuple


@dataclass
class Dataset:
    max_nparams: int
    inputs: np.ndarray
    outputs: np.ndarray


class EvaluationResult(NamedTuple):
    score: float
    optimal_params: np.ndarray
    mse: float


def dataset_evaluator(skeleton: function.Skeleton, arg: Dataset) -> float:
    inputs, outputs = arg.inputs, arg.outputs
    num_input_cols = inputs.shape[1]
    input_args = [inputs[:, i] for i in range(num_input_cols)]

    def loss(params):
        y_pred = skeleton(*input_args, params)
        return np.mean(np.abs(y_pred - outputs) ** 2)

    result = minimize(loss, [1.0] * arg.max_nparams)
    loss_val = result.fun

    if np.isnan(loss_val) or np.isinf(loss_val):
        raise ValueError("loss is inf or nan")

    return float(-loss_val)


def enhanced_dataset_evaluator(skeleton: function.Skeleton, arg: Dataset) -> EvaluationResult:
    inputs, outputs = arg.inputs, arg.outputs
    num_input_cols = inputs.shape[1]
    input_args = [inputs[:, i] for i in range(num_input_cols)]

    def loss(params):
        y_pred = skeleton(*input_args, params)
        return np.mean(np.abs(y_pred - outputs) ** 2)

    result = minimize(loss, [1.0] * arg.max_nparams)
    mse = result.fun

    if np.isnan(mse) or np.isinf(mse):
        raise ValueError("loss is inf or nan")

    return EvaluationResult(
        score=float(-mse),
        optimal_params=result.x,
        mse=mse
    )
