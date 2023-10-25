import os
import time
import cv2
import numpy as np
import cvzone
from ultralytics import YOLO
from utilities.utils import point_in_polygons, draw_roi

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# setting the ROI (polygon) of the frame and loading the video stream
points_polygon = [[[247, 298], [201, 679], [1112, 679], [1020, 294], [248, 296]]]
stream_name = "arquivo.avi"
cap = f'raw-videos/{stream_name}'

# load the model and the COCO class labels our YOLO model was trained on
model = YOLO("models/yolov8x.pt")


def main():
    writer = None
    # pass the frame to the yolov8 model
    for result in model(source=cap, verbose=False, stream=True, show=False, classes=[0, 1, 2, 3, 5, 7],
                        conf=0.3, agnostic_nms=True, iou=0.5):
        start = time.time()
        frame = result.orig_img

        detections = []
        boxes = result.boxes
        for r in boxes:
            # extract the bounding box coordinates, confidence and class of each object
            x1, x2, x3, x4 = map(int, r.xyxy[0])
            class_id = int(r.cls[0])
            score = float(r.conf[0])
            w, h = x3 - x1, x4 - x2

            if points_polygon is not None:
                # Check if the centroid of each object is inside the polygon
                cX = (x1 + x3) / 2
                cY = (x2 + x4) / 2
                if not point_in_polygons((cX, cY), points_polygon):
                    continue

            # Display a rectangle with customized corners.
            cvzone.cornerRect(frame, (x1, x2, w, h), l=10, t=4)
            cvzone.putTextRect(frame,
                               f'{classNames[class_id]}',
                               (max(0, x1), max(35, x2)),
                               scale=0.5, thickness=1, colorR=(224, 182, 90),
                               colorT=(40, 40, 40),
                               font=cv2.FONT_HERSHEY_DUPLEX,
                               offset=5)

        end = time.time()
        print("[INFO] classification time " + str((end - start) * 1000) + "ms")

        # draw roi
        output_frame = frame if points_polygon is None else draw_roi(frame, points_polygon)
        # output_frame = frame

        # Resize the frame to show on the screen
        resized = cv2.resize(frame, (1200, int(output_frame.shape[0] * 1200 / output_frame.shape[1])))

        # save the video with detections
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter('output/adwall03.avi', fourcc, 15, (frame.shape[1], frame.shape[0]), True)
        writer.write(output_frame)

        # show the output
        cv2.imshow('Frame', resized)
        key = cv2.waitKey(1) & 0xFF

        # if key == ord('k'):
        #     cv2.imwrite("yolovB.jpg", output_frame)

        # stop the frame
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
