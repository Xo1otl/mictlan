from dataclasses import dataclass
import inspect
import jax
import jax.numpy as np
import optax
from funsearch import function
from funsearch import profiler

jax.config.update('jax_platform_name', 'cpu')

MAX_NPARAMS = 10


@dataclass
class EvaluatorArg:
    inputs: np.ndarray
    outputs: np.ndarray


def multi_arg_skeleton(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params[0]*x1**2 + params[1]*x2**2 + params[2]*x3**2 + params[3]*x1*x2 + params[4]*x2*x3 + params[5]*x1*x3


def actual_function(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    # めっちゃ複雑な関数
    t = np.tanh(np.sin(x1) * np.exp(-x2**2) + np.cos(x2) * np.log(x3**2 + 1))
    t = np.tanh(t + np.tanh(x1 * x3) * (x1 - x2)) + t
    t = np.tanh(t + (x1**2 - x2**2) * np.sin(x3))
    return t


def lbfgs_evaluator(skeleton: function.Skeleton, arg: EvaluatorArg) -> float:
    inputs = arg.inputs
    outputs = arg.outputs
    x1, x2, x3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]

    def loss_fn(params):
        return np.mean((skeleton(x1, x2, x3, params) - outputs) ** 2)

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
        # 試した感じ10回で大体みつけてくるけど、念のため30回
        body_fn, (init_params, opt_state), None, length=30)

    return float(-loss_fn(final_params))


def adam_evaluator(skeleton: function.Skeleton, arg: EvaluatorArg) -> float:
    x1, x2, x3 = arg.inputs[:, 0], arg.inputs[:, 1], arg.inputs[:, 2]
    targets = arg.outputs

    def loss_fn(params):
        return np.mean((skeleton(x1, x2, x3, params) - targets)**2)
    grad_fn = jax.grad(loss_fn)
    optimizer = optax.adam(3e-3)
    init_params = np.ones(MAX_NPARAMS)
    init_opt_state = optimizer.init(init_params)

    def body_fn(carry, _):
        params, opt_state = carry
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None
    (final_params, _), _ = jax.lax.scan(
        body_fn, (init_params, init_opt_state), None, length=1000)

    return float(-loss_fn(final_params))


def test_py_ast_skeleton():
    src = inspect.getsource(multi_arg_skeleton)
    py_ast_skeleton = function.PyAstSkeleton(src)
    n = 2000
    x1 = np.linspace(-2, 2, n)
    x2 = np.linspace(-1, 1, n)
    x3 = np.linspace(0, 3, n)
    inputs = np.stack([x1, x2, x3], axis=1)
    outputs = actual_function(x1, x2, x3)
    arg = EvaluatorArg(inputs, outputs)
    props = function.DefaultFunctionProps(
        py_ast_skeleton, [arg], lbfgs_evaluator)
    fn = function.DefaultFunction(props)
    fn.use_profiler(profiler.default_fn)
    print(fn.evaluate())


if __name__ == "__main__":
    test_py_ast_skeleton()
