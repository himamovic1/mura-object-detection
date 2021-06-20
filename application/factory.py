from flask import Flask

from config.config import Config
from .navigation import init_site_navigation
from .plugins import init_plugins
from .routes import init_routes


def create_app() -> Flask:
    app_config = Config()
    flask_app = Flask(
        __name__, template_folder=app_config.FLASK_TEMPLATES_PATH, static_folder=app_config.FLASK_STATIC_PATH
    )

    flask_app.config.from_object(app_config)

    init_plugins(flask_app)
    init_routes(flask_app)
    init_site_navigation(flask_app)

    return flask_app
