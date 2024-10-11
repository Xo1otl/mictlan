from infra import ecosystem

HOSTNAME = f'mail.{ecosystem.DOMAIN}'
CONTAINER_NAME = 'mailserver'

def docker_compose():
    from .docker_compose import docker_compose
    return docker_compose