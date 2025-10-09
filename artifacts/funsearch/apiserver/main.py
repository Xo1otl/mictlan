from funsearch import llmsr
from funsearch import datadriven
import numpy as np

# 追加できるパラメータの最大数
MAX_NPARAMS = 1

# 毎回参考にする関数のドキュメント
docstring = r'''
Mathematical function for the tensile modulus of a particle-filler rubber composite.

This function aims to model the relationship between the filler volume fraction (phi)
and the experimentally observed tensile modulus. The evolution should be based on
the Reuss model, which defines the lower bound for the composite's elastic modulus.

The Reuss model ($E_{\text{lower}}$) is given by:
$E_{\text{lower}} = \frac{E_m E_f}{(1-\phi)E_f + \phi E_m}$

The core of this function, to be evolved by FunSearch, **must use** the provided
tensile modulus of the matrix (E_m), the filler (E_f), and the volume fraction (phi).
It is strongly encouraged to incorporate the Reuss model structure. The goal is to
build upon or modify this physical model to better fit the data, potentially using
the optimizable `params`.

Args:
    volume_fraction: A numpy array representing the filler volume fraction (phi).
    params: Array of numeric constants (at most MAX_NPARAMS) to be optimized.
    E_m: Tensile modulus of the matrix (fixed at 4.84).
    E_f: Tensile modulus of the filler (fixed at 117.64).

Return:
    A numpy array for the predicted tensile modulus (E_composite).
'''

# 関数の初期値
equation_src = r'''
def equation(volume_fraction: np.ndarray, params: np.ndarray, E_m=4.84, E_f=117.64) -> np.ndarray:
    phi = volume_fraction

    # The original Reuss model is defined as the starting point.
    E_lower = (E_m * E_f) / ((1 - phi) * E_f + phi * E_m)

    # FunSearch should evolve the following return statement.
    # The goal is to modify this base equation, using the `params` array
    # if necessary, to better match the experimental data.
    return E_lower
'''

# 入力データの配列、複数の入力がある可能性もあるので配列の配列形式
inputs = [[0], [0.09], [0.17], [0.33], [0.44]]

# 関数の出力の想定解(数字)の配列
outputs = [4.84, 5.56, 6.13, 10.13, 14.96]

# 関数の中に書きたくはないが指示として書きたいことをプロンプトコメントとして追加設定できる
prompt_comment = r"""
Discover a mathematical function to predict the tensile modulus of a particle-filler rubber composite.
The function must be based on established physical models.

You are given the following variables:
- phi: The filler volume fraction (the primary input).
- E_m: The tensile modulus of the matrix (4.84).
- E_f: The tensile modulus of the filler (117.64).
- E_composite: The modded Reuss model.

Your task is to evolve the function `E_composite = f(phi, params, E_m, E_f)`.
Start with the provided Reuss model. You can introduce up to MAX_NPARAMS
optimizable parameters (from the `params` array) to modify or extend the Reuss model
to achieve a better fit with the experimental data. For instance, you could try
scaling it, adding terms, or modifying its components. The goal is to find a
physically meaningful improvement to the basic Reuss model.
"""


def main():
    datasets = [datadriven.Dataset(max_nparams=MAX_NPARAMS, inputs=np.array(
        inputs), outputs=np.array(outputs))]

    evolver = llmsr.spawn_evolver_for_mcp(llmsr.EvolverConfigForMCP(
        equation_src=equation_src,
        docstring=docstring,
        evaluation_inputs=datasets,
        evaluator=datadriven.dataset_evaluator,
        prompt_comment=prompt_comment,
        profiler_fn=llmsr.Profiler().profile,
        max_nparams=MAX_NPARAMS,
    ))

    evolver.start()


if __name__ == "__main__":
    main()
