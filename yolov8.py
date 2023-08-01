import os
import time
import cv2
import imutils
import numpy as np
import cvzone
from ultralytics import YOLO
from utilities.utils import point_in_polygons, draw_roi

# setting the ROI (polygon) of the frame and loading the video stream
points_polygon = [[[784, 713], [804, 403], [578, 404], [15, 492], [9, 710], [781, 715]]]
stream = u'......avi'

# load the model and the COCO class labels our YOLO model was trained on
model = YOLO("models/yolov8x.pt")
labelsPath = os.path.sep.join(["coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")



def main():
    writer = None
    # pass the frame to the yolov8 model
    for result in model(source=stream, verbose=False, stream=True, show=False, classes=[0, 1, 2, 3, 5, 7],
                        conf=0.3, agnostic_nms=True):
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

            # Check if the centroid of each object is inside the polygon
            cX = (x1 + x3) / 2
            cY = (x2 + x4) / 2
            if not point_in_polygons((cX, cY), points_polygon):
                continue

            # if class_id == 0:
            #     LABELS[int(class_id)] = 'motorbike'

            # mostra retangulo com corner customizado
            cvzone.cornerRect(frame, (x1, x2, w, h), l=10, t=4)
            cvzone.putTextRect(frame,
                               f'{LABELS[int(class_id)]}',
                               (max(0, x1), max(35, x2)),
                               scale=0.5, thickness=1, colorR=(224, 182, 90),
                               colorT=(40, 40, 40),
                               font=cv2.FONT_HERSHEY_DUPLEX,
                               offset=5)

        end = time.time()
        print("[INFO] classification time " + str((end - start) * 1000) + "ms")

        # draw roi
        # output_frame = draw_roi(frame, points_polygon)
        # resized = imutils.resize(frame, width=1200)

        # save the video with detections
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter('yolov8.avi', fourcc, 20, (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)

        # show the output
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        # if key == ord('k'):
        #     cv2.imwrite("DN5005B.jpg", output_frame)

        # stop the frame
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
