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

app = FastAPI()

# Store historical data
historical_data: List[Dict] = []

# HTML template
html = """
<!DOCTYPE html>
<html>
<head>
    <title>People Counter</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { display: flex; gap: 20px; }
        .video-container { flex: 1; }
        .table-container { flex: 1; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        #videoFeed { max-width: 100%; }
    </style>
</head>
<body>
    <h1>Live People Counter</h1>
    <div class="container">
        <div class="video-container">
            <h2>Live Feed</h2>
            <img id="videoFeed" src="">
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
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            document.getElementById('videoFeed').src = data.frame;
            updateTable(data.history);
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
    
    model = YOLO('yolov8n.pt')
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
            
            # Add count text to frame
            cv2.putText(annotated_frame, f'People: {people_count}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert frame to base64 for sending over WebSocket
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Update historical data
            current_time = datetime.now().strftime("%H:%M:%S")
            historical_data.append({
                "time": current_time,
                "count": people_count
            })
            
            # Keep only last 30 entries
            if len(historical_data) > 30:
                historical_data.pop(0)
            
            # Send data to client
            await websocket.send_json({
                "frame": f"data:image/jpeg;base64,{frame_base64}",
                "history": historical_data
            })
            
            await asyncio.sleep(0.1)  # Limit frame rate
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
