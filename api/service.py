from deepface import DeepFace
import base64
import os, tempfile
import requests

def writeImage(img_base64):
    tmpfl = tempfile.NamedTemporaryFile(delete=False)
    try:
        if img_base64[0:4]=="data":
            img_base64 = img_base64.split(",")[1]
            
        tmpfl.write(base64.b64decode(img_base64))
        return tmpfl.name
    finally:
        tmpfl.close()

def fetchImage(img_url):
    response = requests.get(img_url)
    # 检查请求是否成功
    if response.status_code == 200:
        # 获取图片数据
        image_data = response.content

        tmpfl = tempfile.NamedTemporaryFile(delete=False)
        tmpfl.write(image_data)
        tmpfl.close()
        return tmpfl.name


def represent(img_path, model_name, detector_backend, enforce_detection, align):
    result = {}
    embedding_objs = DeepFace.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    result["results"] = embedding_objs
    return result


def verify(
    img1_path, img2_path, model_name, detector_backend, distance_metric, enforce_detection, align
):
    obj = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )
    return obj


def analyze(img_path, actions, detector_backend, enforce_detection, align):
    result = {}
    demographies = DeepFace.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    result["results"] = demographies
    return result
