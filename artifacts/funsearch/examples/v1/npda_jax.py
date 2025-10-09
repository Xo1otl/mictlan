"""
jax では係数同定ができないけど、いろいろ試した結果をのこす
"""

from funsearch import function
from funsearch import llmsr
from dataclasses import dataclass
import jaxopt
import jax
import jax.numpy as np
import optax
import pandas as pd

# evaluate で gpu 使ってみたら時々バグって ollama が止まるし、計算量的に普通に cpu のほうが速い
# jax.config.update('jax_platform_name', 'cpu') # adam の場合は gpu のほうが速い
jax.config.update('jax_enable_x64', True)  # 精度あげても found_equation が収束できない

MAX_NPARAMS = 10


@dataclass
class EvaluatorArg:
    inputs: np.ndarray
    outputs: np.ndarray


def found_equation(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for shg efficiency

    Args:
        width: A numpy array representing periodic domain width
        wavelength: A numpy array representing wavelength.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing shg efficiency as the result of applying the mathematical function to the inputs.
    """
    # Scaling factor for the domain width. This accounts for variations in the poling period.
    dw_scaling = params[0] + params[1] / wavelength

    # Calculate the effective domain width.
    effective_width = width * dw_scaling

    # Total length of the QPM device. This is determined by the number of domains and the effective width.
    L = params[2] * effective_width

    # Grating vector. This is related to the poling period and the wavelength.
    k_g = params[3]

    # Phase mismatch calculation. This is the difference between the phase of the fundamental wave and the phase of the second harmonic wave.
    delta_k = k_g + params[4] - np.pi / effective_width

    # Argument for the sinc^2 function.
    arg = delta_k * L / 2

    # SHG efficiency calculation. The sinc^2 function is the key element.
    # The constant factor (params[5]) scales the overall efficiency.
    # The denominator (arg**2 + params[6]**2) introduces a damping factor that broadens the peak.
    efficiency = params[5] * L**2 * np.sin(arg)**2 / (arg**2 + params[6]**2)

    # Damping factor to account for losses and imperfections.
    # This is a polynomial function of the phase mismatch.
    damping_factor = 2 + params[7] * arg + params[8] * arg**2

    # Apply the damping factor to the efficiency.
    efficiency = efficiency / damping_factor

    return efficiency


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


# FIXME: なぜか jax の bfgs だと収束しないし時間もめっちゃかかる
def bfgs_evaluator(skeleton: function.Skeleton[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], arg: EvaluatorArg) -> float:
    inputs = arg.inputs
    outputs = arg.outputs
    width, wavelength = inputs[:, 0], inputs[:, 1]

    def loss_fn(params: np.ndarray):
        return np.mean((skeleton(width, wavelength, params) - outputs) ** 2)

    # FIXME: なんか大量のINFOとWARNINGが出る
    solver = jaxopt.BFGS(fun=loss_fn, verbose=False)
    init_params = np.ones(MAX_NPARAMS) * 0.1
    results = solver.run(init_params=init_params)
    final_params = results.params
    final_loss = loss_fn(final_params)
    return float(-final_loss)


# adam やってみたけど found_equation は収束しない
def adam_evaluator(skeleton: function.Skeleton[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], arg: EvaluatorArg) -> float:
    x1, x2 = arg.inputs[:, 0], arg.inputs[:, 1]
    targets = arg.outputs

    def loss_fn(params):
        return np.mean((skeleton(x1, x2, params) - targets)**2)
    grad_fn = jax.grad(loss_fn)
    optimizer = optax.adam(3e-4)
    init_params = np.ones(MAX_NPARAMS)
    init_opt_state = optimizer.init(init_params)

    def body_fn(carry, _):
        params, opt_state = carry
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None
    (final_params, _), _ = jax.lax.scan(
        body_fn, (init_params, init_opt_state), None, length=10000)

    return float(-loss_fn(final_params))


# Callable と同じ型の指定方法で skeleton の型を指定する
def lbfgs_evaluator(skeleton: function.Skeleton[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], arg: EvaluatorArg) -> float:
    inputs = arg.inputs
    outputs = arg.outputs
    width, wavelength = inputs[:, 0], inputs[:, 1]

    def loss_fn(params):
        return np.mean((skeleton(width, wavelength, params) - outputs) ** 2)

    solver = optax.lbfgs()
    init_params = np.ones(MAX_NPARAMS) * 0.1
    # spicy が見つけた初期値を使うとより小さい誤差に収束できることがわかる
    # init_params = np.array([-2.81395596e+00,  4.27215799e+00, 1.85079628e+01,  3.29352756e-01,
    #                         3.29352697e-01,  3.66807525e-04,  2.72828390e-02, - 3.05278855e+01,
    #                         1.34336994e+02,  1.00000000e+00])
    opt_state = solver.init(init_params)

    value_and_grad = optax.value_and_grad_from_state(loss_fn)

    def body_fn(carry, _):
        params, opt_state = carry
        loss_value, grad = value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(
            grad, opt_state, params, value=loss_value, grad=grad, value_fn=loss_fn)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None

    (final_params, _), _ = jax.lax.scan(
        body_fn, (init_params, opt_state), None, length=100)

    loss = float(-loss_fn(final_params))
    if np.isnan(loss) or np.isinf(loss):
        raise ValueError("loss is inf or nan")
    else:
        return loss


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
        loss = adam_evaluator(found_equation, input)
        losses.append(loss)
    print(f"losses: {losses}")


def main():
    inputs = load_inputs()
    test_evaluate(inputs)

#     prompt_comment = """
# Find the mathematical function skeleton that represents SHG efficiency in vertical Quasi-Phase Matching devices, given domain width and wavelength.
# The final efficiency expression is expected to be proportional to the square of a sinc-like function involving terms derived from width and wavelength.
# """  # prompt_comment の mathmatical function skeleton という用語とても大切、これがないと llm が params の存在を忘れて細かい値を設定し始める

#     evolver = llmsr.spawn_evolver(llmsr.EvolverConfig(
#         equation=equation,
#         evaluation_inputs=inputs,
#         evaluator=lbfgs_evaluator,
#         prompt_comment=prompt_comment,
#         profiler_fn=llmsr.Profiler().profile,
#     ))

#     evolver.start()


if __name__ == "__main__":
    main()
