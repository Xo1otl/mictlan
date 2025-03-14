from syuron import shg
from .use_material import *


def analyze(params: shg.Params) -> shg.EffTensor:
    return shg.analyze(params, use_material, shg.solve_ncme)
