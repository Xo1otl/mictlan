PORT = 6379

def compose():
    from .redis import redis_service
    return redis_service
