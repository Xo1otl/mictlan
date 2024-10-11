docker_compose = {
    'services': {
        'answer': {
            'container_name': 'answer',
            'build': {
                'dockerfile': 'Dockerfile',
            },
            'ports': [
                '9080:80',
            ],
            'volumes': [
                'answer-data:/data',
            ],
        },
    },
    'volumes': {
        'answer-data': None,
    },
}
