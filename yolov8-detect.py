import os
import cv2
import imutils
import numpy as np
from ultralytics import YOLO
from utilities.utils import point_in_polygons, draw_roi

# setting the ROI (polygon) of the frame and loading the video stream
# points_polygon = [[[418, 506], [588, 436], [1397, 577], [1015, 686], [762, 583], [321, 713], [416, 508]]]
# stream = u'rtmp://rtmp01.datavisiooh.com:1935/calles06'
# points_polygon = [[[554, 908], [1312, 1437], [2092, 893], [1363, 490], [552, 908]]]
# stream = u'rtmp://rtmp01.datavisiooh.com:1935/calles_t21
points_polygon = [[[1228, 445], [665, 1017], [98, 913], [680, 385], [1230, 443]]]
stream = u'rtmp://rtmp01.datavisiooh.com:1935/calles_t17'

cap = cv2.VideoCapture(stream)

# load the model and the COCO class labels our YOLO model was trained on
model = YOLO("models/yolov8x.pt")
labelsPath = os.path.sep.join(["coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")
writer = None

# set the confidence
detection_threshold = 0.1

while True:
    ret, frame = cap.read()

    # pass the frame to the yolov8 detector
    results = model(frame)

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
            if score < detection_threshold:
                continue

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[int(class_id)]]
            cv2.rectangle(frame, (x1, x2), (x3, x4), color, 2)
            # text = "{}: {:.4f}".format(LABELS[int(class_id)], score)
            # cv2.putText(frame, text, (x1, x2 - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

    # draw roi
    output_frame = draw_roi(frame, points_polygon)
    resized = imutils.resize(output_frame, width=1200)

    # # save the video with detections
    # if writer is None:
    #     fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #     writer = cv2.VideoWriter('torre-t8.avi', fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    # writer.write(output_frame)

    # show the output
    cv2.imshow('Frame', resized)
    key = cv2.waitKey(1) & 0xFF

    # stop the frame
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()