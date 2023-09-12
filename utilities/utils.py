from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import cv2
import numpy as np


def point_in_polygons(points, list_polygons):
    point = Point(points)

    for i in list_polygons:
        polygon = Polygon(i)
        if polygon.contains(point):
            return True
    return False


def draw_roi(frame, polygons):
    frame_overlay = frame.copy()
    for polygon in polygons:
        cv2.fillPoly(
            frame_overlay,
            np.array([polygon], dtype=np.int32),
            (0, 127, 127)
        )

    alpha = 0.2
    output_frame = cv2.addWeighted(frame_overlay, alpha, frame, 1 - alpha, 0)
    return output_frame
