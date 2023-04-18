import os
import time
import cv2
import imutils
import numpy as np
from ultralytics import YOLO
from utilities.utils import point_in_polygons, draw_roi
from tracker.centroid import CentroidTracker

# setting the ROI (polygon) of the frame and loading the video stream
points_polygon = [[[500, 1409], [2028, 1393], [1615, 452], [981, 443], [501, 1411]]]
stream = u'rtmp://rtmp01.datavisiooh.com:1935/prime_gt03'
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
threshold = 0.5

while True:
    ret, frame = cap.read()

    start = time.time()
    # pass the frame to the yolov8 detector
    results = model(source=frame, verbose=False)
    end = time.time()
    print("[INFO] classification time " + str((end - start) * 1000) + "ms")

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            # extract the bounding box coordinates, confidence and class of each object
            x1, x2, x3, x4, score, class_id = r
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
            if score < confidenceLevel:
                continue

            # update our list of bounding box coordinates,
            # confidences, and class IDs
            boxes.append([x1, x2, x3, x4])
            confidences.append(float(score))
            classIDs.append(class_id)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidenceLevel,
                            threshold)
    rects = []
    classDetected_list = []
    confDegree_list = []

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():

            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if LABELS[classIDs[i]] not in ["person", "car", "motorbike", "bus", "bicycle", "truck"]:
                continue

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                       confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            rects.append([x, y, w, h])
            classDetected_list.append(LABELS[classIDs[i]])
            confDegree_list.append(confidences[i])
            print(i)
            exit(0)

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects, classDetected_list, confDegree_list)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = f"ID {objectID}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),  #############
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), - 1)

    # # save the video with detections
    # if writer is None:
    #     fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #     writer = cv2.VideoWriter('torre-t15.avi', fourcc, 30, (frame.shape[1], frame.shape[0]), True)
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