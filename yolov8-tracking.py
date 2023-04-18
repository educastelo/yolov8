import os
import time
import cv2
import imutils
import numpy as np
from ultralytics import YOLO
from utilities.utils import point_in_polygons, draw_roi
from tracker.centroid import CentroidTracker

# setting the ROI (polygon) of the frame and loading the video stream
points_polygon = [[[2308, 1429], [1555, 1429], [1307, 610], [1723, 633], [2310, 1427]]]
stream = u'rtmp://rtmp01.datavisiooh.com:1935/cartel_VER5001A'
cap = cv2.VideoCapture(stream)

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=1, maxDistance=400)
trackers = []
trackableObjects = {}

# load the model and the COCO class labels our YOLO model was trained on
model = YOLO("models/yolov8s.pt")
labelsPath = os.path.sep.join(["coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")
writer = None

# set the confidence
confidenceLevel = 0.1
threshold = 0.3

while True:
    ret, frame = cap.read()

    start = time.time()

    # pass the frame to the yolov8 detector
    results = model(source=frame, verbose=False)
    end = time.time()
    print("[INFO] classification time " + str((end - start) * 1000) + "ms")

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            # extract the bounding box coordinates, confidence and class of each object
            x1, x2, x3, x4, confidence, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            x3 = int(x3)
            x4 = int(x4)
            class_id = int(class_id)

            # Check if the centroid of each object is inside the polygon
            cX = (x1 + x3) / 2
            cY = (x2 + x4) / 2

            test_polygon = point_in_polygons((cX, cY), points_polygon)
            if not test_polygon:
                continue

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence < confidenceLevel:
                continue


            if LABELS[class_id] not in ["person", "car", "motorbike", "bus", "bicycle", "truck"]:
                continue

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[class_id]]
            cv2.rectangle(frame, (x1, x2), (x3, x4), color, 2)
            text = "{}: {:.4f}".format(LABELS[class_id],  ################
                                       confidence)
            cv2.putText(frame, text, (x1, x2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # update our centroid tracker using the computed set of bounding
            # box rectangles
            objects = ct.update([x1, x2, x3, x4], LABELS[class_id], confidence)

            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = f"ID {objectID}"
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),  #############
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), - 1)

    ## save the video with detections
    # if writer is None:
    #     fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #     writer = cv2.VideoWriter('traffic-yolov8.avi', fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    # writer.write(frame)

    # draw roi
    output_frame = draw_roi(frame, points_polygon)
    resized = imutils.resize(output_frame, width=1200)

    # show the output
    cv2.imshow('Frame', resized)
    key = cv2.waitKey(1) & 0xFF

    # stop the frame
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()