from flask import Blueprint, request
import service
import os

blueprint = Blueprint("routes", __name__)


@blueprint.route("/")
def home():
    return "<h1>Welcome to DeepFace API!</h1>"


@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_base64 = input_args.get("img")
    img_url = input_args.get("img_url")
    if img_base64 is None and img_url is None:
        return {"message": "you must pass img input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    align = input_args.get("align", True)

    img_path = ""
    if img_base64 is None:
        img_path = service.fetchImage(img_url)
    else:
        img_path = service.writeImage(img_base64)
    
    obj = service.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    return obj


@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img1_base64 = input_args.get("img1")
    img1_url = input_args.get("img1_url")
    img2_base64 = input_args.get("img2")
    img2_url = input_args.get("img2_url")

    if img1_base64 is None and img1_url is None:
        return {"message": "you must pass img1 input"}

    if img2_base64 is None and img2_url is None:
        return {"message": "you must pass img2 input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    distance_metric = input_args.get("distance_metric", "cosine")
    align = input_args.get("align", True)

    img1_path = ""
    if img1_base64 is None:
        img1_path = service.fetchImage(img1_url)
    else:
        img1_path = service.writeImage(img1_base64)
    img2_path = ""
    if img2_base64 is None:
        img2_path = service.fetchImage(img2_url)
    else:
        img2_path = service.writeImage(img2_base64)

    verification = service.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )
    verification["verified"] = str(verification["verified"])
    os.unlink(img1_path)
    os.unlink(img2_path)

    return verification


@blueprint.route("/analyze", methods=["POST"])
def analyze():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_base64 = input_args.get("img")
    img_url = input_args.get("img_url")
    if img_base64 is None and img_url is None:
        return {"message": "you must pass img input"}

    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    align = input_args.get("align", True)
    actions = input_args.get("actions", ["age", "gender", "emotion", "race"])

    img_path = ""
    if img_base64 is None:
        img_path = service.fetchImage(img_url)
    else:
        img_path = service.writeImage(img_base64)

    demographies = service.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    os.unlink(img_path)
    return demographies
