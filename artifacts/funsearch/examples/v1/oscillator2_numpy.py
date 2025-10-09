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


# FIXME: 評価関数の中で使いまわす値をわざわざ引数から渡ってくるようにしているが、インスタンスに紐づける等の工夫でこのような回りくどい設計にする必要がなくなる
def scipy_evaluator(skeleton: function.Skeleton[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray], arg: EvaluatorArg) -> float:
    """ Evaluate the equation on data observations."""

    # Load data observations
    inputs, outputs = arg.inputs, arg.outputs
    t, x, v = inputs[:, 0], inputs[:, 1], inputs[:, 2]

    def loss(params):
        y_pred = skeleton(t, x, v, params)
        return np.mean((y_pred - outputs) ** 2)

    result = minimize(loss, [1.0]*MAX_NPARAMS, method='BFGS')
    loss = result.fun

    return float(-loss)  # type: ignore


def equation_v2(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    omega0 = params[0]  # natural frequency
    gamma = params[1]   # damping ratio
    F0 = params[2]     # amplitude of driving force
    omega_d = params[3]  # angular frequency of driving force
    alpha = params[4]   # nonlinear term coefficient for position
    beta = params[5]    # nonlinear term coefficient for velocity
    delta = params[6]   # additional damping term coefficient
    eta = params[7]     # coupling parameter between position and velocity
    phi = params[8]     # phase shift in driving force
    chi = params[9]     # additional nonlinear term coefficient
    dvdt = -omega0**2 * x - 2 * gamma * omega0 * v - delta * v + F0 * \
        np.cos(omega_d * t + phi) + alpha * (x**2) + \
        beta * (v**2) + eta * x * v + chi * (x**3)
    return dvdt


def equation_v3(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    dv = params[0] * np.sin(t) - params[1] * x - params[2] * v + params[3] * x**2 - params[4] * v**3 + params[5] * np.sin(params[6] * t) - params[7] * x * v + params[8] * np.cos(params[9] * t)
    return dv

def equation_v4(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    dv = params[0] * np.sin(params[1] * t) - params[2] * x - params[3] * v + params[4] * x**2 - params[5] * v**3 + params[6] * np.cos(params[7] * t) * v - params[8] * x * v + params[9] * np.exp(-params[3] * (x**2 + v**2)) * np.sin(t) + params[1] * np.tanh(x) # Included tanh(x)
    return dv


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for acceleration in a damped nonlinear oscillator

    Args:
        t: A numpy array representing time.
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
    """
    dv = params[0] * t + params[1] * x + params[2] * v + params[3]
    return dv


def load_inputs():
    # 必要なデータのロード
    evaluation_inputs = []
    # 論文の方では探索では train.csv しか使ってなかった。スコアパターンとかかいとるからてっきり全部計算するのかおもた
    data_files = ['train.csv', 'test_id.csv', 'test_ood.csv']
    for data_file in data_files:
        df = pd.read_csv(
            f'/workspaces/mictlan/research/funsearch/data/oscillator2/{data_file}')
        data = np.array(df)
        inputs = data[:, :-1]
        outputs = data[:, -1].reshape(-1)
        evaluation_inputs.append(EvaluatorArg(inputs, outputs))
    return evaluation_inputs


def test_evaluate(inputs):
    losses = []
    for input in inputs:
        loss = scipy_evaluator(equation_v4, input)
        losses.append(loss)
    print(f"losses: {losses}")


def main():
    inputs = load_inputs()
    test_evaluate(inputs)
    return

    prompt_comment = """
Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on time, position, and velocity. 
"""  # prompt_comment の mathmatical function skeleton という用語とても大切、これがないと llm が params の存在を忘れて細かい値を設定し始める

    evolver = llmsr.spawn_evolver(llmsr.EvolverConfig(
        equation=equation,
        evaluation_inputs=inputs,
        evaluator=scipy_evaluator,
        prompt_comment=prompt_comment,
        profiler_fn=llmsr.Profiler().profile,
        num_parallel=5
    ))

    evolver.start()


if __name__ == "__main__":
    main()
