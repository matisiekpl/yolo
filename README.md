# Real-time People Counter

A real-time people counting application that uses YOLOv8 for person detection and FastAPI for serving a web interface. The application provides live video feed with person detection, historical data tracking, and visual analytics.

![Screenshot](screenshot.png)

## Features

- Real-time person detection using YOLOv8
- Live video feed with bounding boxes and count display
- Historical data tracking with timestamps
- Interactive line chart showing people count over time
- Responsive web interface

> This project uses [uv](https://github.com/astral-sh/uv) package manager

## Usage

1. Run the application:
```bash
uv run main.py
# or
uv run legacy.py # for running legacy WebSocket-based system
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

## Evaluation

You can evaluate YOLO model on sample images stored in `eval/`.

Example:
```bash
uv run predict.py eval/eval1.png
```

Enable annotated output:
```bash
uv run predict.py eval/eval1.png --show
```

## API Endpoints

The application exposes the following REST API endpoints:

### GET /
- Returns the main HTML interface

### GET /healthz 
- Returns system health metrics including CPU and memory usage

### GET /data
- Returns historical people count data
- Response: Array of objects containing:
  - `timestamp`: Unix timestamp in milliseconds
  - `count`: Number of people detected

### GET /image
- Returns the latest captured frame with detection annotations
- Response: PNG image

### POST /purge
- Clears all historical data
