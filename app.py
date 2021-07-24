from flask import Flask

from application.factory import create_app
from config.config import Config

if __name__ == "__main__":
    application: Flask = create_app(app_config=Config())
    application.run()
