from .env import *

PORT = 7700


def compose():
    from .compose import compose
    return compose
