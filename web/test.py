import json
import os

from flask import Flask, request
from werkzeug.utils import secure_filename

from util.findContours import cutImg
from util.predict import PredictModel
from util.readimg import ReadImgandCvtBinaryNoBlur
import numpy as np

app = Flask(__name__)


@app.route("/parse/path", methods=["POST"])
def parse_path():
    postForm = request.form
    filepath = postForm.get("path")
    _, res = cutImg(filepath)
    data = np.asarray(res).reshape([-1, 784])
    result = p.predict(data)
    retjson = {"code": 0, "result": result}
    return json.dumps(retjson, ensure_ascii=False)


@app.route("/parse/file", methods=["POST"])
def parse_file():
    file = request.files["image"]
    if file:
        filename = secure_filename(file.filename)
        path = os.path.join("../static", filename)
        file.save(path)
        _, res = cutImg(path)
        data = np.asarray(res).reshape([-1, 784])
        result = p.predict(data)
        return json.dumps({"code": 0, "result": result})
    return json.dumps({"code": 1, "result": "no image"})


if __name__ == "__main__":
    p = PredictModel()
    app.run()
