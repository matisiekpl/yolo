from ultralytics import YOLO
import cv2
import time
from datetime import datetime
import csv
import os
import threading
from fastapi import FastAPI
from typing import List, Dict
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stop_threads = threading.Event()


@app.get("/data")
async def get_data() -> List[Dict[str, int]]:
    data = []
    try:
        with open("log.csv", "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(
                    {"timestamp": int(row["timestamp"]), "count": int(row["count"])}
                )
    except FileNotFoundError:
        return []
    return data


def detection_thread():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    csv_filename = "log.csv"
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["timestamp", "count"])

        try:
            while not stop_threads.is_set():
                loop_start_time = time.time()

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

                # Save annotated frame
                cv2.imwrite("log.png", annotated_frame)

                # Log to CSV with Unix timestamp in milliseconds
                unix_timestamp_ms = int(time.time() * 1000)
                csv_writer.writerow([unix_timestamp_ms, people_count])
                csvfile.flush()  # Ensure data is written to disk

                # Calculate time to sleep to maintain 1 second interval
                processing_time = time.time() - loop_start_time
                sleep_time = max(
                    0, 1.0 - processing_time
                )  # Ensure non-negative sleep time
                time.sleep(sleep_time)

        except Exception as e:
            print(f"Detection thread error: {e}")
        finally:
            cap.release()


if __name__ == "__main__":
    # Start detection thread
    detection_thread = threading.Thread(target=detection_thread)
    detection_thread.start()

    try:
        # Start FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\nStopping the application...")
    finally:
        stop_threads.set()  # Signal thread to stop
        detection_thread.join()  # Wait for detection thread to finish
