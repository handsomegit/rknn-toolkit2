from fastapi import FastAPI, UploadFile, File, Request, HTTPException
import numpy as np
import cv2
from yolov8_infer import setup_model, post_process, CLASSES, draw
from py_utils.coco_utils import COCO_test_helper
import argparse
import time
import uuid
import gc

app = FastAPI()

# 最大图片数量限制
MAX_IMAGES_PER_REQUEST = 10

model, platform = setup_model(argparse.Namespace(
    model_path='../model/yolov8.rknn',
    target='rk3588',
    device_id=None
))

co_helper = COCO_test_helper(enable_letter_box=True)


def preprocess_image(img_src):
    """单张图片预处理"""
    img = co_helper.letter_box(img_src.copy(), new_shape=(640, 640), pad_color=(0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 增加 batch 维度 (1, 640, 640, 3)
    return np.expand_dims(img, 0)


@app.get("/v1/vision/ping")
async def ping():
    return {"status": "ok", "message": "YOLOv8 API is running"}


@app.post("/v1/vision/detection")
async def detect(request: Request):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    if not model:
        raise HTTPException(503, "Model not loaded")

    form = await request.form()
    image_files: list[tuple[int, UploadFile]] = []

    # 解析上传图片
    for key, value in form.items():
        if not key.lower().startswith("image"):
            continue
        if not hasattr(value, "filename"):
            continue

        num_str = key.lower().replace("image", "").replace("[]", "").strip("_-")
        if not num_str.isdigit():
            # 如果是 'image' 没有数字，默认为 0
            if key.lower() == "image":
                num_str = "0"
            else:
                continue

        image_index = int(num_str)
        image_files.append((image_index, value))

    if not image_files:
        return {"success": False, "requestId": request_id, "message": "No images"}

    if len(image_files) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(400, f"Too many images, max is {MAX_IMAGES_PER_REQUEST}")

    image_files.sort(key=lambda x: x[0])

    # ----------------------------
    # 读取并预处理所有图片
    # ----------------------------
    imgs = []
    img_file_map = {}  # image_index -> UploadFile
    for image_index, file in image_files:
        data = await file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        imgs.append((image_index, img))
        img_file_map[image_index] = file  # 保存原始 file 对象

    if not imgs:
        return {"success": False, "requestId": request_id, "message": "All images invalid"}

    # ----------------------------
    # 逐图推理 (RKNN 模型不支持 batch)
    # ----------------------------
    predictions = []
    images_info = []
    total_infer_ms = 0

    for idx, (image_index, img) in enumerate(imgs):
        # 预处理单张图片
        input_data = preprocess_image(img)

        # 单张推理
        t0 = time.time()
        outputs = model.run([input_data])
        infer_ms = int((time.time() - t0) * 1000)
        total_infer_ms += infer_ms

        # 后处理
        boxes, classes, scores = post_process(outputs)

        count = 0
        if boxes is not None and classes is not None and scores is not None:
            # 将检测框从 letter_box 尺寸映射回原图尺寸
            real_boxes = co_helper.get_real_box(boxes)
            for i in range(len(scores)):
                box = real_boxes[i].astype(int)
                predictions.append({
                    "image_index": image_index,
                    "label": CLASSES[int(classes[i])],
                    "confidence": float(scores[i]),
                    "x_min": int(box[0]),
                    "y_min": int(box[1]),
                    "x_max": int(box[2]),
                    "y_max": int(box[3])
                })
                count += 1

        file = img_file_map[image_index]
        images_info.append({
            "image_index": image_index,
            "filename": file.filename,
            "success": True,
            "count": count,
            "inferenceMs": infer_ms
        })

        # 及时释放
        del img, input_data, boxes, classes, scores

    gc.collect()

    return {
        "success": True,
        "requestId": request_id,
        "processMs": int((time.time() - start_time) * 1000),
        "image_count": len(image_files),
        "object_count": len(predictions),
        "predictions": predictions,
        "images": images_info
    }
