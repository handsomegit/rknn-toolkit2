from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from yolov8_infer import setup_model, post_process, CLASSES, draw
from py_utils.coco_utils import COCO_test_helper
import argparse

app = FastAPI()

model, platform = setup_model(argparse.Namespace(
    model_path='../model/yolov8.rknn',
    target='rk3588',
    device_id=None
))

co_helper = COCO_test_helper(enable_letter_box=True)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_data = await file.read()
    img_array = np.frombuffer(image_data, np.uint8)
    img_src = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img_src is None:
        return {"success": False, "error": "Invalid image"}

    img = co_helper.letter_box(img_src.copy(), new_shape=(640, 640), pad_color=(0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_data = img
    outputs = model.run([input_data])
    boxes, classes, scores = post_process(outputs)

    result = []
    if boxes is not None:
        real_boxes = co_helper.get_real_box(boxes)
        for i in range(len(scores)):
            box = real_boxes[i].astype(int)
            result.append({
                "label": CLASSES[int(classes[i])],
                "confidence": float(scores[i]),
                "x_min": int(box[0]),
                "y_min": int(box[1]),
                "x_max": int(box[2]),
                "y_max": int(box[3])
            })

    return {"success": True, "objects": result}
