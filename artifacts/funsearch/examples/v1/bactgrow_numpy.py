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

def equation_v2(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # idやoodを使うと係数が発見できない、train.csvで発見した係数を手動で設定すれば他のデータセットでも精度を確認可能
    # params = np.array([4.79695157e+03,  9.97774840e-01,  3.09969435e+01,  7.10961033e+00,
    #                    1.83658124e+01,  8.80574701e+03, -9.35937532e-01, -2.23846922e+01,
    #                    -9.24085707e+00,  2.87205527e+03])
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


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for bacterial growth rate

    Args:
        b: A numpy array representing observations of population density of the bacterial species.
        s: A numpy array representing observations of substrate concentration.
        temp: A numpy array representing observations of temperature.
        pH: A numpy array representing observations of pH level.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing bacterial growth rate as the result of applying the mathematical function to the inputs.
    """
    return params[0] * b + params[1] * s + params[2] * temp + params[3] * pH + params[4]


def scipy_evaluator(skeleton: function.Skeleton[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray], arg: EvaluatorArg) -> float:
    """ Evaluate the equation on data observations."""

    # Load data observations
    inputs, outputs = arg.inputs, arg.outputs
    b, s, temp, pH = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]

    def loss(params):
        y_pred = skeleton(b, s, temp, pH, params)
        return np.mean((y_pred - outputs) ** 2)

    result = minimize(loss, [1.0]*MAX_NPARAMS, method='BFGS')
    print(result.x)
    loss = float(-result.fun)  # type: ignore

    # nan や inf の場合にエラーを投げるのを忘れたらあかん (他のexampleで忘れてるやつある、、、)
    if np.isnan(loss) or np.isinf(loss):  # type: ignore
        raise ValueError("loss is inf or nan")
    else:
        return loss  # type: ignore


def load_inputs():
    # 必要なデータのロード
    evaluation_inputs = []
    # 論文の方では探索では train.csv しか使ってなかった。スコアパターンとかかいとるからてっきり全部計算するのかおもた
    data_files = ['train.csv', 'test_id.csv', 'test_ood.csv']
    for data_file in data_files:
        df = pd.read_csv(
            f'/workspaces/mictlan/research/funsearch/data/bactgrow/{data_file}')
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

    prompt_comment = """
Find the mathematical function skeleton that represents E. Coli bacterial growth rate, given data on population density, substrate concentration, temperature, and pH level.
"""  # prompt_comment の mathmatical function skeleton という用語とても大切、これがないと llm が params の存在を忘れて細かい値を設定し始める

    evolver = llmsr.spawn_evolver(llmsr.EvolverConfig(
        equation=equation,
        evaluation_inputs=inputs,
        evaluator=scipy_evaluator,
        prompt_comment=prompt_comment,
        profiler_fn=llmsr.Profiler().profile,
        num_parallel=5,
        reset_period=10 * 60
    ))

    evolver.start()


if __name__ == "__main__":
    main()
