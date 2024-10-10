DOMAIN = 'mictlan.site'


def generate_docker_compose():
    from .docker_compose import generate_docker_compose
    generate_docker_compose()
