from . import *
from infra.db import rdb

services = {
    'lemmy': {
        'image': 'dessalines/lemmy:0.19.8',
        'hostname': 'lemmy',
        'environment': {
            'RUST_LOG': 'warn'
        },
        'volumes': [
            './lemmy.hjson:/config/config.hjson:Z'
        ],
        'depends_on': [
            'postgres',
            'pictrs'
        ]
    },
    'lemmy-ui': {
        'image': 'dessalines/lemmy-ui:0.19.8',
        'environment': {
            'LEMMY_UI_LEMMY_INTERNAL_HOST': 'lemmy:8536',
            'LEMMY_UI_LEMMY_EXTERNAL_HOST': 'lemmy.ml',
            'LEMMY_UI_HTTPS': 'false'
        },
        'depends_on': [
            'lemmy'
        ],
    },
    'pictrs': {
        'image': 'asonix/pictrs:0.5.16',
        'hostname': 'pictrs',
        'environment': {
            'PICTRS_OPENTELEMETRY_URL': 'http://otel:4137',
            'PICTRS__SERVER__API_KEY': f'{DBPASSWORD}',
            'PICTRS__MEDIA__VIDEO__VIDEO_CODEC': 'vp9',
            'PICTRS__MEDIA__ANIMATION__MAX_WIDTH': '256',
            'PICTRS__MEDIA__ANIMATION__MAX_HEIGHT': '256',
            'PICTRS__MEDIA__ANIMATION__MAX_FRAME_COUNT': '400'
        },
        'user': '991:991',
    }
}
