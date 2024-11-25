PORT = 9080
SERVICE_NAME = 'ossekai'


def compose():
    from .compose import compose
    return compose
