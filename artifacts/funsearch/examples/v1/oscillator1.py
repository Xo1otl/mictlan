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


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for acceleration in a damped nonlinear oscillator

    Args:
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
    """
    return params[0] * x + params[1] * v + params[2]


def found_equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    acceleration = params[0] + params[1] * v - params[2] * \
        x + params[3] * x * v - params[4] * x**2 * np.sin(x)
    return acceleration


def lbfgs_evaluator(skeleton: function.Skeleton[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], arg: EvaluatorArg) -> float:
    inputs = arg.inputs
    outputs = arg.outputs
    x, v = inputs[:, 0], inputs[:, 1]

    def loss_fn(params):
        return np.mean((skeleton(x, v, params) - outputs) ** 2)

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
    data_files = ['train.csv', 'test_id.csv', 'test_ood.csv']
    for data_file in data_files:
        df = pd.read_csv(
            f'/workspaces/mictlan/research/funsearch/data/oscillator1/{data_file}')
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
    test_evaluate(inputs)
    return

    prompt_comment = """
Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity.
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
