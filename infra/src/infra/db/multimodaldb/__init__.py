PORT = "8000"
SERVICE_NAME = "multimodaldb"


def compose():
    from .compose import compose
    return compose
