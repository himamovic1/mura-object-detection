from flask import Flask

from app_plugins import init_plugins
from app_routes import init_routes
from config.config import Config


def create_app() -> Flask:
    app_config = Config()
    flask_app = Flask(
        __name__,
        template_folder=app_config.FLASK_TEMPLATES_PATH,
        static_folder=app_config.FLASK_STATIC_PATH
    )

    flask_app.config.from_object(app_config)

    init_plugins(flask_app)
    init_routes(flask_app)

    return flask_app


if __name__ == "__main__":
    application: Flask = create_app()
    application.run()
