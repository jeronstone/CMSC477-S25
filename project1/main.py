import cv2
import numpy as np
import time
import traceback
from queue import Empty
from robomaster import robot
from robomaster import camera
from ApriltagDetector import *
from ibvs import get_ibvs_speeds

def draw_detections(frame, detections):
    for detection in detections:
        pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)

        frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        top_left = tuple(pts[0][0])  # First corner
        top_right = tuple(pts[1][0])  # Second corner
        bottom_right = tuple(pts[2][0])  # Third corner
        bottom_left = tuple(pts[3][0])  # Fourth corner
        cv2.line(frame, top_left, bottom_right, color=(0, 0, 255), thickness=2)
        cv2.line(frame, top_right, bottom_left, color=(0, 0, 255), thickness=2)

if __name__ == '__main__':
    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YR")
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel
    marker_size_m = 0.153 # Size of the AprilTag in meters
    apriltag = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)

    offsets = [
        (33, (1, 0)), 
        (33, (2, 0)),
        (33, (2, 1))
    ]
    curr = 0

    while (True):
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)

        detections = apriltag.find_tags(gray)

        if curr < len(offsets):

            if len(detections) < 1:
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)

            for detect in detections:
                if detect.tag_id == offsets[curr][0]:
                    res, deltas = get_ibvs_speeds(detect, offsets[curr][1])

                    error = np.linalg.norm(deltas)
                    #print(f'Error: {error}')

                    if error > 0.05:
                        ep_chassis.drive_speed(x=res.item(1), y=0, z=res.item(0), timeout=5)
                    else:
                        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
                        curr += 1
                        print('curr ++')
                        break

        else:
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
        
        draw_detections(img, detections)
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
            break
