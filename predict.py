import os
from ultralytics import YOLO
import cv2
import sys
import argparse

MODEL_NAME = os.getenv('MODEL_NAME', 'yolov8n.pt')

def count_people(image_path, show=False):
    model = YOLO(MODEL_NAME)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}", file=sys.stderr)
        sys.exit(1)

    results = model(image, classes=[0], verbose=False)
    people_count = len(results[0].boxes)
    print(people_count)

    if show:
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
        cv2.imshow("People Counter", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Count number of people in an image.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument(
        "--show", action="store_true", help="Display the image with detections"
    )
    args = parser.parse_args()

    count_people(args.image_path, args.show)


if __name__ == "__main__":
    main()
