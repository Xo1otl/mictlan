PORT = 27017
SERVICE_NAME = 'mongo'


def compose():
    from .compose import compose
    return compose
