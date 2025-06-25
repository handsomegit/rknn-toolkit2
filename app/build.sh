docker build -t yolov8-rknn-api .



docker run --rm -it --net=host --privileged   -v /dev:/dev   -p 8000:8000   yolov8-rknn-api
