authelia_service = {
    'container_name': 'authelia',
    'image': 'authelia/authelia:latest',
    'volumes': ['./configuration.yml:/config/configuration.yml'],
}
