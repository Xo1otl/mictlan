from syuron import shg
from .use_material import *
from .train_and_eval import *
from .baysian_optim_and_eval import *


def analyze(params: shg.Params) -> shg.EffTensor:
    return shg.analyze(params, use_material, shg.solve_ncme_npda)
