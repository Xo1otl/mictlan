DBNAME = 'affine'
USERNAME = 'affine'
PORTS = [3010, 5555]


def compose():
    from .affine import affine_service
    return {
        'services': {
            'affine': affine_service
        }
    }
