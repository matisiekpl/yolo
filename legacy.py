import os
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import time
import json
from datetime import datetime
import asyncio
from typing import List, Dict
import base64
import numpy as np

MODEL_NAME = os.getenv('MODEL_NAME', 'yolov8n.pt')

app = FastAPI()

historical_data: List[Dict] = []

html = """
<!DOCTYPE html>
<html>
<head>
    <title>People Counter</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { display: flex; flex-direction: column; gap: 20px; }
        .row { display: flex; gap: 20px; }
        .video-container { flex: 1; }
        .table-container { flex: 1; }
        .chart-container { 
            flex: 1; 
            height: 200px;
            position: relative;
            margin-bottom: 20px;
        }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        #videoFeed { max-width: 100%; }
        #peopleChart {
            width: 100% !important;
            height: 200px !important;
            max-height: 200px !important;
        }
    </style>
</head>
<body>
    <h1>Live People Counter</h1>
    <div class="container">
        <div class="row">
            <div class="video-container">
                <h2>Live Feed</h2>
                <img id="videoFeed" src="">
            </div>
            <div class="chart-container">
                <h2>People Count Over Time</h2>
                <canvas id="peopleChart"></canvas>
            </div>
        </div>
        <div class="table-container">
            <h2>Historical Data</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody id="historyTable">
                </tbody>
            </table>
        </div>
    </div>
    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        let chart;

        function initChart() {
            const ctx = document.getElementById('peopleChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Number of People',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    layout: {
                        padding: {
                            top: 10,
                            bottom: 10
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                stepSize: 1
                            }
                        }
                    }
                }
            });
        }

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            document.getElementById('videoFeed').src = data.frame;
            updateTable(data.history);
            updateChart(data.history);
        };
        
        function updateTable(history) {
            const tbody = document.getElementById('historyTable');
            tbody.innerHTML = '';
            history.forEach(entry => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${entry.time}</td>
                    <td>${entry.count}</td>
                `;
                tbody.appendChild(row);
            });
        }

        function updateChart(history) {
            chart.data.labels = history.map(entry => entry.time);
            chart.data.datasets[0].data = history.map(entry => entry.count);
            chart.update();
        }

        initChart();
    </script>
</body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, classes=[0])
            people_count = len(results[0].boxes)
            annotated_frame = results[0].plot()

            cv2.putText(
                annotated_frame,
                f"People: {people_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            _, buffer = cv2.imencode(".jpg", annotated_frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            current_time = datetime.now().strftime("%H:%M:%S")
            historical_data.append({"time": current_time, "count": people_count})

            if len(historical_data) > 30:
                historical_data.pop(0)

            await websocket.send_json(
                {
                    "frame": f"data:image/jpeg;base64,{frame_base64}",
                    "history": historical_data,
                }
            )

            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
