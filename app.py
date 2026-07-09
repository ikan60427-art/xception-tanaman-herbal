import os

from flask import (
    Flask,
    render_template,
    request
)

from werkzeug.utils import secure_filename

from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template(
            "index.html",
            error="Silakan pilih gambar terlebih dahulu."
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "index.html",
            error="Silakan pilih gambar."
        )

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    label, confidence = predict_image(filepath)

    return render_template(

        "index.html",

        image=filepath,

        label=label.capitalize(),

        confidence=round(confidence,2)

    )


if __name__ == "__main__":

    app.run(
        debug=True
    )