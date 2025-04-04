from ultralytics import YOLO
import cv2
import time
from robomaster import robot
from robomaster import camera
from ibvs_controller import *
from yolotest import clamp, sub_data_handler

ROBOT_X_VELOCITY_MIN = -0.1
ROBOT_X_VELOCITY_MAX = 0.1
ROBOT_Y_VELOCITY_MIN = -0.15
ROBOT_Y_VELOCITY_MAX = 0.15
ROBOT_Z_VELOCITY_MIN = -1.0
ROBOT_Z_VELOCITY_MAX = 1.0
ROBOT_Z_POSITION_MIN = -50
ROBOT_Z_POSITION_MAX = 20
ROBOT_Z_ANGULAR_VELOCITY_MIN = -0.5
ROBOT_Z_ANGULAR_VELOCITY_MAX = 0.5

robot_z_position = 0 # keeps track of arm position

model = YOLO(r"..\runs\detect\train\weights\best.pt")

# Use vid instead of ep_camera to use your laptop's webcam
# vid = cv2.VideoCapture(0)


ep_robot = robot.Robot()
ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YR")
ep_chassis = ep_robot.chassis
ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
ep_arm = ep_robot.robotic_arm
ep_arm.moveto(x=200, y=0).wait_for_completed()
ep_gripper = ep_robot.gripper

# Start printing the gripper position
# ep_arm.sub_position(freq=5, callback=sub_data_handler)

controller = IBVS_Controller(control_mode='4xyzy', interaction_mode='desired', num_pts=4)
controller.set_lambda_matrix([2.0, 10.0, 1.0, 1000.0]) # robot y velocity; robot z velocity; robot x velocity; robot z angular velocity
controller.set_desired_points([(-0.16, 0.2, 0.215), (0.16, 0.2, 0.215), (-0.16, 0.9, 0.215), (0.16, 0.9, 0.215)])

while True:
    # ret, frame = vid.read()
    try:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
    except:
        frame = None
        continue
    if frame is not None:
        start = time.time()
        if model.predictor:
            model.predictor.args.verbose = False
        result = model.predict(source=frame, show=False)[0]

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
ep_robot.close()