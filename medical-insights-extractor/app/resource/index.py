import aiohttp_jinja2
from aiohttp import web


class Index(web.View):
    @aiohttp_jinja2.template('index.html')
    async def get(self):
        return {}