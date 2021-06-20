from flask import Flask
from flask_bootstrap import Bootstrap


def init_plugins(app: Flask) -> None:
    """Add new routes to application level"""
    Bootstrap(app)
