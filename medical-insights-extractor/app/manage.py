import os
from aiohttp import web

from app.config import CONFIG
from app.common.utils import load_config, get_logger


def create_app():
    app = web.Application(client_max_size=1024 * 1024 * 1024)
    config_name = os.getenv('ENVIRONMENT', 'development')
    app.config = load_config(CONFIG[config_name])
    logger = get_logger()
    print(f'Starting server for: [{app.config.ENVIRONMENT}]')
    return app, logger
