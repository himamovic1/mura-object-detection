import os

from flask import Flask, request, flash, render_template, url_for

from util.image import validate_image_type, generate_filename


def init_routes(app: Flask) -> None:
    """ Add new routes to application level """

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            uploaded_file = request.files['file']

            if not validate_image_type(uploaded_file.filename):
                flash("Image not of valid type")
                return render_template('index.html')

            img_name = generate_filename(uploaded_file.filename)
            image_path = os.path.join(app.config["FLASK_STATIC_PATH"], img_name)
            uploaded_file.save(image_path)

            # Prediction should be done in this point
            # class_name = model.get_prediction(image_path)

            return render_template('result.html', result={
                'class_name': "class_name",
                'image_path': url_for("static", filename=img_name),
            })

        return render_template('index.html')
