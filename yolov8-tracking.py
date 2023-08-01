import os
import time
import cv2
import imutils
import numpy as np
import cvzone
from ultralytics import YOLO
from utilities.utils import point_in_polygons, draw_roi
from tracker.centroid import CentroidTracker

# setting the ROI (polygon) of the frame and loading the video stream
points_polygon = [[[352, 260], [1110, 524], [998, 717], [167, 716], [348, 258]]]
stream = 'rtmp://rtmp....'

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=1, maxDistance=400)
trackers = []
trackableObjects = {}

# load the model and the COCO class labels our YOLO model was trained on
model = YOLO("models/yolov8m.pt")
labelsPath = os.path.sep.join(["coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")
writer = None


def main():
    # Stream is online, so proceed with reading the frames
    for result in model(source=stream, verbose=False, max_det=200, stream=True, show=False,
                        conf=0.3, agnostic_nms=True, classes=[0, 1, 2, 3, 5, 7]):
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

            # Check if the centroid of each object is inside the polygon
            cX = (x1 + x3) / 2
            cY = (x2 + x4) / 2
            if not point_in_polygons((cX, cY), points_polygon):
                continue

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[class_id]]
            cv2.rectangle(frame, (x1, x2), (x3, x4), color, 2)
            text = "{}: {:.4f}".format(LABELS[class_id],
                                       score)
            cv2.putText(frame, text, (x1, x2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            rects.append([x1, x2, x3, x4])
            classDetected_list.append(LABELS[class_id])
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
        output_frame = draw_roi(frame, points_polygon)
        resized = imutils.resize(output_frame, width=1200)

        # save the video with detections
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter('yolov8....avi', fourcc, 10, (frame.shape[1], frame.shape[0]), True)
        writer.write(output_frame)

        # show the output
        cv2.imshow('Frame', resized)
        key = cv2.waitKey(1) & 0xFF

        # stop the frame
        if key == ord('q'):
            exit(1)


if __name__ == '__main__':
    main()
