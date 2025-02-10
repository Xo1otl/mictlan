PORT = 6379
REDIS_URL = f'redis://redis:{PORT}'


def compose():
    from .redis import redis_service
    return redis_service
