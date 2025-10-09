from funsearch import function
from funsearch import llmsr
from dataclasses import dataclass
from scipy.optimize import minimize, basinhopping
import numpy as np
# import jax.numpy as np # scipy.optimize.minimize は jax.numpy だと動かないことを試してみた
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
    width, wavelength = inputs[:, 0], inputs[:, 1]

    def loss(params):
        y_pred = skeleton(width, wavelength, params)
        return np.mean((y_pred - outputs) ** 2)

    result = basinhopping(loss, [1.0] * MAX_NPARAMS, disp=True, minimizer_kwargs={"method": "BFGS"})  # これでも係数が見つからない
    # result = minimize(loss, [1.0]*MAX_NPARAMS, method='BFGS')
    loss = result.fun

    return float(-loss)  # type: ignore


def unoptimizable_equation(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:
    # ここをコメントアウトしてevaluateすれば、摂動項を追加することで同定できた係数で目的関数がちゃんと最適化されていることがわかる
    # params = np.array([8.95695069e-01, -4.14636143e+00, -3.95280532e-03,  9.50745281e+00,
    #                    4.74926578e-01,  3.06783245e+01, -2.70970610e+00,  3.61013288e+00,
    #                    -9.52100818e-03,  1.00000000e+00])
    k_mismatch = 2 * np.pi / wavelength - 2 * np.pi / (params[0] * width)
    phase_shift = params[1]
    broadening_factor = params[2]
    sinc_component = np.sinc(
        (k_mismatch + phase_shift) / (2 * broadening_factor))
    shg_efficiency = params[3] * sinc_component**2
    return shg_efficiency


def found_equation(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:
    k_mismatch = 2 * np.pi / wavelength - 2 * np.pi / (params[0] * width)
    phase_shift = params[1]
    broadening_factor = params[2]
    sinc_component = np.sinc(
        (k_mismatch + phase_shift) / (2 * broadening_factor))
    non_linear_effect = params[3] * sinc_component**2
    higher_order_interactions = params[4] * \
        wavelength**(-params[5]) + params[6] * width**(-params[7])
    phase_matching_effect = params[8] * k_mismatch**2
    shg_efficiency = non_linear_effect + \
        higher_order_interactions + phase_matching_effect
    return shg_efficiency


def equation(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for shg efficiency

    Args:
        width: A numpy array representing periodic domain width
        wavelength: A numpy array representing wavelength.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing shg efficiency as the result of applying the mathematical function to the inputs.
    """
    num_domains = params[0]
    return num_domains * width + params[1] * wavelength + params[2]


def load_inputs():
    # 必要なデータのロード
    evaluation_inputs = []
    # 論文の方では探索では train.csv しか使ってなかった。スコアパターンとかかいとるからてっきり全部計算するのかおもた
    data_files = ['train.csv', 'test_id.csv', 'test_ood.csv']
    for data_file in data_files:
        df = pd.read_csv(
            f'/workspaces/mictlan/research/funsearch/data/npda/{data_file}')
        data = np.array(df)
        inputs = data[:, :-1]
        outputs = data[:, -1].reshape(-1)
        evaluation_inputs.append(EvaluatorArg(inputs, outputs))
    return evaluation_inputs


def test_evaluate(inputs):
    losses = []
    for input in inputs:
        loss = scipy_evaluator(unoptimizable_equation, input)
        losses.append(loss)
    print(f"losses: {losses}")


def main():
    inputs = load_inputs()
    # test_evaluate(inputs)
    # return

    # FIXME: qwen2.5-coder は SHG というワードを知らないので Second Harmonic Generation という正式名称を伝えるべきだった、なくても発見できたのはラッキーだけど今後気をつける
    prompt_comment = """
Find the mathematical function skeleton that represents SHG efficiency in vertical Quasi-Phase Matching devices, given domain width and wavelength.
The final efficiency expression is expected to be proportional to the square of a sinc-like function involving terms derived from width and wavelength.
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
