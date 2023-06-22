import os
import time
import cv2
import numpy as np
import cvzone
from ultralytics import YOLO
from utilities.utils import point_in_polygons, draw_roi

# setting the ROI (polygon) of the frame and loading the video stream
points_polygon = [[[297, 107], [306, 312], [926, 253], [899, 93], [298, 108]]]
stream = 'rtmp://rtmp03.datavisiooh.com:1935/cartel_DN5004A'
cap = cv2.VideoCapture(stream)
fpsReader = cvzone.FPS()

# load the model and the COCO class labels our YOLO model was trained on
model = YOLO("models/yolov8m.pt")
labelsPath = os.path.sep.join(["coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")
writer = None

while True:
    ret, frame = cap.read()

    fps, frame = fpsReader.update(frame, pos=(50, 80), color=(0, 255, 0), scale=5, thickness=5)

    # start = time.time()
    # pass the frame to the yolov8 detector
    results = model(source=frame, verbose=True, max_det=200, stream=True, show=False,
                    conf=0.3, agnostic_nms=True, classes=[0, 1, 2, 3, 5, 7])
    # end = time.time()
    # tempo_total = (end - start) * 1000
    print(f"FPS Stream 3: {fps}")
    # print(f"[INFO] classification time stream 3: {tempo_total} ms")

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

#             # draw a bounding box rectangle and label on the frame
#             color = [int(c) for c in COLORS[int(class_id)]]
#             cv2.rectangle(frame, (x1, x2), (x3, x4), color, 6)
#             text = "{}: {:.2f}".format(LABELS[int(class_id)], score)
#             cv2.putText(frame, text, (x1, x2 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
#
#     # draw roi
#     output_frame = draw_roi(frame, points_polygon)
#     resized = imutils.resize(output_frame, width=1200)
#
#     # # save the video with detections
#     # if writer is None:
#     #     fourcc = cv2.VideoWriter_fourcc(*"XVID")
#     #     writer = cv2.VideoWriter('yolov8.avi', fourcc, 15, (frame.shape[1], frame.shape[0]), True)
#     # writer.write(output_frame)
#
#     # show the output
#     cv2.imshow('Frame', resized)
#     key = cv2.waitKey(1) & 0xFF
#
#     if key == ord('k'):
#         cv2.imwrite("screenshot.jpg", frame)
#
#     # stop the frame
#     if key == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
