import os

from flask import Flask, request, flash, render_template, url_for
from werkzeug.exceptions import HTTPException

from config.config import Config
from detector.inference import detect_and_mark_objects
from util.image import validate_image_type, generate_filename


def init_routes(app: Flask) -> None:
    """Add new routes to application level"""

    @app.errorhandler(HTTPException)
    def handle_exception(e):
        return render_template(
            "error.html",
            error={
                "status_code": e.code,
                "name": e.name,
                "description": e.description,
                "stacktrace": e.original_exception,
            },
        )

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            uploaded_file = request.files["file"]

            if not validate_image_type(uploaded_file.filename):
                flash("Image not of valid type")
                return render_template("index.html")

            img_name = generate_filename(uploaded_file.filename)
            image_path = os.path.join(app.config["FLASK_STATIC_PATH"], img_name)
            uploaded_file.save(image_path)

            # Prediction should be done in this point
            detection_result = detect_and_mark_objects(image_path=image_path, app_config=Config())

            return render_template(
                "result.html",
                result={
                    "scores": detection_result,
                    "image_path": url_for("static", filename=img_name),
                },
            )

        return render_template("index.html")
