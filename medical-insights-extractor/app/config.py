CONFIG = {
    "production": "app.config.ProductionConfig",
    "staging": "app.config.StagingConfig",
    "development": "app.config.DevelopmentConfig"
}


class BaseConfig(object):
    APP_VERSION = '1.0.0'


class DevelopmentConfig(BaseConfig):
    ENVIRONMENT = 'development'
    INPUT_FOLDER = 'app/static/'
