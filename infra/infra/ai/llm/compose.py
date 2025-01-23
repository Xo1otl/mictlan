from . import *

STORAGE_DIR = '/app/server/storage'
OLLAMA_BASE_PATH = f'http://{OLLAMA_SERVICE_NAME}:{OLLAMA_PORT}'

compose = {
    'services': {
        OLLAMA_SERVICE_NAME: {
            'image': 'ollama/ollama:latest',
            'container_name': OLLAMA_SERVICE_NAME,
            'volumes': ['ollama:/root/.ollama'],
            'ports': [f'{OLLAMA_PORT}:{OLLAMA_PORT}'],
            'environment': [
                'OLLAMA_NUM_PARALLEL=16',  # デフォルトだと4しか無い
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
        },
        'anythingllm': {
            # 2025/1/13時点での最新版はフロントエンドが動かないエラーあり、1.3かdevを使用
            'image': 'mintplexlabs/anythingllm:dev',
            'container_name': 'anythingllm',
            'ports': [f'{ANYTHINGLLM_PORT}:3001'],
            'cap_add': ['SYS_ADMIN'],
            'environment': [
                f'STORAGE_DIR={STORAGE_DIR}',
                f'JWT_SECRET={JWT_SECRET}',
                'LLM_PROVIDER=ollama',
                f'OLLAMA_BASE_PATH={OLLAMA_BASE_PATH}',
                'OLLAMA_MODEL_PREF=deepseek-r1:32b',
                'OLLAMA_MODEL_TOKEN_LIMIT=4096',
                'EMBEDDING_ENGINE=ollama',
                f'EMBEDDING_BASE_PATH={OLLAMA_BASE_PATH}',
                'EMBEDDING_MODEL_PREF=snowflake-arctic-embed2:latest',
                'EMBEDDING_MODEL_MAX_CHUNK_LENGTH=8192',
                'VECTOR_DB=lancedb',
                'WHISPER_PROVIDER=local',
                'TTS_PROVIDER=native',
                'PASSWORDMINCHAR=8',
            ],
            'volumes': [f'anythingllm_storage:{STORAGE_DIR}'],
            'depends_on': ['ollama']
        }
    },
    'volumes': {
        'ollama': None,
        'anythingllm_storage': None
    }
}
