from flask import Flask
from flask_nav import Nav
from flask_nav.elements import Navbar, View


def init_site_navigation(app: Flask) -> None:
    """Add new navbar to application level"""
    nav = Nav()

    nav.register_element(
        "top",
        Navbar(
            "Bone X-Ray Object Detection",
            View("Home", "index"),
        ),
    )

    nav.init_app(app)
