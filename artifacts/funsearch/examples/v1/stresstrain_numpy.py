from funsearch import function
from funsearch import llmsr
from dataclasses import dataclass
from scipy.optimize import minimize
import numpy as np
import pandas as pd

MAX_NPARAMS = 10


@dataclass
class EvaluatorArg:
    inputs: np.ndarray
    outputs: np.ndarray


def scipy_evaluator(skeleton: function.Skeleton[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], arg: EvaluatorArg) -> float:
    """ Evaluate the equation on data observations."""

    # Load data observations
    inputs, outputs = arg.inputs, arg.outputs
    strain, temp = inputs[:, 0], inputs[:, 1]

    def loss(params):
        y_pred = skeleton(strain, temp, params)
        return np.mean((y_pred - outputs) ** 2)

    result = minimize(loss, [1.0]*MAX_NPARAMS, method='BFGS')
    loss = result.fun

    return float(-loss)  # type: ignore


def equation_v2(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    stress = (params[0] * strain + params[1] * temp + params[2] * strain**2 + params[3] * temp**2 + params[4] * strain * temp + params[5] * strain**3
              + params[6] * temp**3 + params[7] * strain**2 * temp + params[8] * strain * temp**2) / (1 + params[9] * strain)
    return stress


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for stress in Aluminium rod

    Args:
        strain: A numpy array representing observations of strain.
        temp: A numpy array representing observations of temperature.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing stress as the result of applying the mathematical function to the inputs.
    """
    return params[0] * strain + params[1] * temp


def load_inputs():
    # 必要なデータのロード
    evaluation_inputs = []
    # 論文の方では探索では train.csv しか使ってなかった。スコアパターンとかかいとるからてっきり全部計算するのかおもた
    data_files = ['train.csv', 'test_id.csv', 'test_ood.csv']
    for data_file in data_files:
        df = pd.read_csv(
            f'/workspaces/mictlan/research/funsearch/data/stressstrain/{data_file}')
        data = np.array(df)
        inputs = data[:, :-1]
        outputs = data[:, -1].reshape(-1)
        evaluation_inputs.append(EvaluatorArg(inputs, outputs))
    return evaluation_inputs


def test_evaluate(inputs):
    losses = []
    for input in inputs:
        loss = scipy_evaluator(equation_v2, input)
        losses.append(loss)
    print(f"losses: {losses}")


def main():
    inputs = load_inputs()
    test_evaluate(inputs)
    return

    prompt_comment = """
Find the mathematical function skeleton that represents stress, given data on strain and temperature in an Aluminium rod for both elastic and plastic regions.
"""  # prompt_comment の mathmatical function skeleton という用語とても大切、これがないと llm が params の存在を忘れて細かい値を設定し始める

    evolver = llmsr.spawn_evolver(llmsr.EvolverConfig(
        equation=equation,
        evaluation_inputs=inputs,
        evaluator=scipy_evaluator,
        prompt_comment=prompt_comment,
        profiler_fn=llmsr.Profiler().profile,
    ))

    evolver.start()


if __name__ == "__main__":
    main()
