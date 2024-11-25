from . import *

command = [
    "/bin/bash",
    "-c",
    "cd /app/ComfyUI && . venv/bin/activate && python main.py --listen 0.0.0.0 --port 8188 & cd /app/fluxgym && . venv/bin/activate && python app.py"
]

compose = {
    'services': {
        'imagegen': {
            'build': '.',
            'ports': [
                f'{COMFYUI_PORT}:{COMFYUI_PORT}',
                f'{FLUXGYM_PORT}:{FLUXGYM_PORT}'
            ],
            'command': command,
            'volumes': [
                'comfyui_output:/app/ComfyUI/output',
                'comfyui_models:/app/ComfyUI/models',
                'comfyui_custom_nodes:/app/ComfyUI/custom_nodes',
                'fluxgym_outputs:/app/fluxgym/outputs',
                'fluxgym_models:/app/fluxgym/models',
                'fluxgym_datasets:/app/fluxgym/datasets',
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
            },
        }
    },
    'volumes': {
        'comfyui_output': {},
        'comfyui_models': {},
        'comfyui_custom_nodes': {},
        'fluxgym_outputs': {},
        'fluxgym_models': {},
        'fluxgym_datasets': {}
    }
}
