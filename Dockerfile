FROM rknn-toolkit2:2.3.2



COPY rknn_model_zoo-v2.3.2-2025-04-09 /workspace/rknn_model_zoo
COPY yolov8n.onnx  /workspace/rknn_model_zoo/examples/yolov8/model

WORKDIR /workspace/rknn_model_zoo/examples/yolov8/python

RUN python3 convert.py ../model/yolov8n.onnx rk3588


COPY rknn-toolkit2-v2.3.2-2025-04-09/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib64/librknnrt.so
COPY rknn-toolkit2-v2.3.2-2025-04-09/rknpu2/runtime/Linux/librknn_api/include/* /usr/include/
COPY rknn-toolkit2-v2.3.2-2025-04-09/rknpu2/runtime/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/

CMD ["python3", "yolov8.py", "--model_path", "../model/yolov8.rknn", "--img_save", "--target", "rk3588"]

# docker build -t harbor.scet.com.cn/scet/yolov8-rknn-api:latest .
