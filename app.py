from flask import Flask

from application.factory import create_app

if __name__ == "__main__":
    application: Flask = create_app()
    application.run()
