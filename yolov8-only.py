import os
import time
import cv2
import imutils
import numpy as np
from ultralytics import YOLO
from utilities.utils import point_in_polygons, draw_roi

# setting the ROI (polygon) of the frame and loading the video stream
points_polygon = [[[2, 213], [521, 203], [1278, 508], [1278, 718], [2, 716], [2, 213]]]
stream = u"rtmp://rtmp0......."

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

    # pass the frame to the yolov8 detector
    for result in model(source=stream, verbose=False, max_det=500, stream=True, show=False, classes=[0, 1, 2, 3, 5, 7],
                        conf=0.1, agnostic_nms=True):
        start = time.time()
        # frame = result.plot(boxes=False, labels=False)
        frame = result.orig_img

        detections = []
        # for r in result.boxes.data.tolist():
        boxes = result.boxes
        for r in boxes:
            # extract the bounding box coordinates, confidence and class of each object
            x1, x2, x3, x4 = map(int, r.xyxy[0])
            class_id = int(r.cls[0])
            score = float(r.conf[0])

            # Check if the centroid of each object is inside the polygon
            cX = (x1 + x3) / 2
            cY = (x2 + x4) / 2
            if not point_in_polygons((cX, cY), points_polygon):
                continue

            # # draw a bounding box rectangle and label on the frame
            # color = [int(c) for c in COLORS[int(class_id)]]
            # cv2.rectangle(frame, (x1, x2), (x3, x4), color, 4)
            # text = "{}: {:.2f}".format(LABELS[int(class_id)], score)
            # cv2.putText(frame, text, (x1, x2 - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 4)

        end = time.time()
        print("[INFO] classification time " + str((end - start) * 1000) + "ms")

        # draw roi
        output_frame = draw_roi(frame, points_polygon)
        resized = imutils.resize(output_frame, width=1500)
        # resized = imutils.resize(frame, width=1200)

        # # save the video with detections
        # if writer is None:
        #     fourcc = cv2.VideoWriter_fourcc(*"XVID")
        #     writer = cv2.VideoWriter('yolov8.avi', fourcc, 20, (frame.shape[1], frame.shape[0]), True)
        # writer.write(output_frame)

        # show the output
        cv2.imshow('Frame', resized)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('k'):
            cv2.imwrite("cartel_STGO5001A.jpg", frame)

        # stop the frame
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
