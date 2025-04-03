from infra import akinator
from workspace import path

entrypoint = """
#!/bin/bash
cd /app/akinator/akinator/ui
pip install poetry
poetry config virtualenvs.create false
poetry install 
streamlit run app.py --server.address 0.0.0.0
"""

compose = {
    'services': {
        'akinator': {
            'image': 'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime',
            'container_name': 'akinator',
            'ports': ['8501:8501'],
            'depends_on': ['mysql', 'redis'],
            'volumes': [
                f'{path.Path("apps/akinator").rel2(path.Path(__file__).dir())}:/app/akinator',
            ],
            'environment': [
                f'MYSQL_USER={akinator.MYSQL_USER}',
                f'MYSQL_PASSWORD={akinator.MYSQL_PASSWORD}',
                f'MYSQL_DB={akinator.MYSQL_DB}',
            ],
            'restart': 'unless-stopped',
            'command': [
                '/bin/bash',
                '-c',
                f'echo \'{entrypoint}\' > /entrypoint.sh && chmod +x /entrypoint.sh && /entrypoint.sh'
            ],
            'deploy': {
                'resources': {
                    'reservations': {
                        'devices': [{
                            'driver': 'nvidia',
                            'capabilities': ['gpu'],
                            'count': 'all'
                        }]
                    }
                }
            }
        }
    },
}
