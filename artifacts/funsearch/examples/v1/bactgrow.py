from funsearch import function
from funsearch import llmsr
from dataclasses import dataclass
import jax
import jax.numpy as np
import optax
import pandas as pd

jax.config.update('jax_platform_name', 'cpu')

MAX_NPARAMS = 10


@dataclass
class EvaluatorArg:
    inputs: np.ndarray
    outputs: np.ndarray


def found_equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    mu_max = params[0]
    Ks = params[1]
    b_sat = params[2]
    Q10 = params[3]
    T_ref = params[4]
    mu_ref = params[5]
    pH_effect = params[6]
    sigma_pH_base = params[7]
    alpha = params[8]
    beta = params[9]
    # Enhanced Monod model for substrate inhibition, including cooperative interactions and substrate saturation effects
    mu_substrate = (mu_max * s) / ((Ks + beta * np.power(s, 2))
                                   * (1 + alpha * b / b_sat))
    # Improved Hill function with logistic transformation for population density effect to better model competition dynamics at high densities
    mu_population = (np.log(1 + alpha * (b / b_sat))) / np.log(1 + alpha)
    # Temperature adaptation enhanced with Q10 factor and Gaussian term, more accurately accounting for thermal stress effects
    T_effect = Q10 ** ((temp - T_ref) / 10.0) * \
        np.exp(-0.05 * (np.power((temp - 37.0), 2)))
    mu_temp = mu_ref * T_effect
    # pH effect modeled as a Gaussian function with adaptive sigma_pH, including higher-order polynomial terms for precise non-linear effects at extreme pH values
    sigma_adaptive = sigma_pH_base + 0.04 * np.abs(pH - 7.0) + 0.025 * np.power(
        (pH - 7.0), 2) + 0.0017 * np.power((pH - 7.0), 3) + 0.0005 * np.power((pH - 7.0), 4)
    mu_pH = mu_temp * np.exp(-pH_effect * ((pH - 7.0) / sigma_adaptive) ** 2)
    # Combined growth rate factors with advanced stability mechanisms, ensuring substrate influence and effective clipping
    return np.clip(mu_substrate * mu_population * mu_pH, 1e-10, mu_max)


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


def lbfgs_evaluator(skeleton: function.Skeleton[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray], arg: EvaluatorArg) -> float:
    inputs = arg.inputs
    outputs = arg.outputs
    b, s, temp, pH = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]

    def loss_fn(params):
        return np.mean((skeleton(b, s, temp, pH, params) - outputs) ** 2)

    solver = optax.lbfgs()
    init_params = np.ones(MAX_NPARAMS)
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
        body_fn, (init_params, opt_state), None, length=30)

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
            f'/workspaces/mictlan/research/funsearch/data/bactgrow/{data_file}')
        data = np.array(df)
        inputs = data[:, :-1]
        outputs = data[:, -1].reshape(-1)
        evaluation_inputs.append(EvaluatorArg(inputs, outputs))
    return evaluation_inputs


def test_evaluate(inputs):
    losses = []
    for input in inputs:
        loss = lbfgs_evaluator(found_equation, input)
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
        evaluator=lbfgs_evaluator,
        prompt_comment=prompt_comment,
        profiler_fn=llmsr.Profiler().profile,
    ))

    evolver.start()


if __name__ == "__main__":
    main()
