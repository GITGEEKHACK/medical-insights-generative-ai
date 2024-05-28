"""gunicorn WSGI server configuration."""
from os import environ
bind = '0.0.0.0:' + environ.get('PORT', '8083')
max_requests = 1000
worker_class = 'aiohttp.worker.GunicornWebWorker'
workers = 2
timeout = 3600