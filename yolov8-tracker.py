import os
import cv2
import cvzone
from ultralytics import YOLO
from utilities.utils import point_in_polygons, draw_roi
from tracker.centroidtracker import CentroidTracker

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
points_polygon = [[[664, 208], [827, 103], [990, 184], [825, 333], [666, 209]]]
stream_name = "arquivo.avi"
cap = f'raw-videos/{stream_name}'

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=1, maxDistance=200)

# load the model and the COCO class labels our YOLO model was trained on
model = YOLO("models/yolov8x.pt")  # escolher o modelo de acordo com a necessidade


def main():
    writer = None
    # Stream is online, so proceed with reading the frames
    for result in model(source=cap, verbose=False, max_det=200, stream=True, show=False,
                        conf=0.1, agnostic_nms=True, classes=[0, 1, 2, 3, 5, 7], iou=0.5):
        frame = result.orig_img
        boxes = result.boxes
        rects = []
        classDetected_list = []
        confDegree_list = []

        for r in boxes:
            # Extract the bounding box coordinates, confidence, and class of each object
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

            # draw a bounding box rectangle and label on the frame
            cvzone.cornerRect(frame, (x1, x2, w, h), l=10, t=4)
            cvzone.putTextRect(frame,
                               f'{classNames[class_id]}',
                               (max(0, x1), max(35, x2)),
                               scale=0.5, thickness=1, colorR=(224, 182, 90),
                               colorT=(40, 40, 40),
                               font=cv2.FONT_HERSHEY_DUPLEX,
                               offset=5)

            rects.append([x1, x2, x3, x4])
            classDetected_list.append(classNames[class_id])
            confDegree_list.append(score)

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = ct.update(rects, classDetected_list, confDegree_list)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), - 1)

        # draw roi
        output_frame = frame if points_polygon is None else draw_roi(frame, points_polygon)

        # Resize the frame to show on the screen
        resized = cv2.resize(output_frame, (1200, int(output_frame.shape[0] * 1200 / output_frame.shape[1])))

        # # Uncomment to save the output video
        # if writer is None:
        #     fourcc = cv2.VideoWriter_fourcc(*"XVID")
        #     writer = cv2.VideoWriter(f'output/{stream_name}', fourcc, 10, (frame.shape[1], frame.shape[0]), True)
        # writer.write(output_frame)

        # show the output
        cv2.imshow('Frame', resized)
        key = cv2.waitKey(1) & 0xFF

        # stop the frame
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
