DOMAIN = 'mictlan.site'


# 遅延ロードしないと循環参照になる
def generate_docker_compose():
    from .docker_compose import generate_docker_compose
    generate_docker_compose()
