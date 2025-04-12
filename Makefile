run:
	MODEL_NAME=yolov8n_ncnn_model uv run main.py

legacy:
	MODEL_NAME=yolov8n_ncnn_model uv run legacy.py

predict:
	MODEL_NAME=yolov8n_ncnn_model uv run predict.py