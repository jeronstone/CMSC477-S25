from enum import Enum
import math
from ultralytics import YOLO
import cv2
import time
import numpy as np
from robomaster import robot
from robomaster import camera
from ibvs_controller import *

def clamp(val, min_val, max_val):
    if val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    else:
        return val
    
def sub_data_handler(sub_info):
    pos_x, pos_y = sub_info

    # Fix the integer overflow in pos_y
    if pos_y > 2**31 - 1:
        pos_y = pos_y - 2**32

    # You can use these values to confirm the robot arm is where it is supposed to be
    # It is also usable for determine the right setpoints to send to "moveto" commands
    # print("Robotic Arm: pos x:{0}, pos y:{1}".format(pos_x, pos_y))

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

ERR_THRESH = 0.125

robot_z_position = 0 # keeps track of arm position

model = YOLO(r"..\runs\detect\train\weights\best.pt")

# Use vid instead of ep_camera to use your laptop's webcam
# vid = cv2.VideoCapture(0)

class State(Enum):
    START_2B_2M = 1,
    MOVE_BLOCK_ON_TARGET_1 = 2,
    GRIP_PICKUP = 3,
    MOVE_EMPTY_1 = 4,
    GRIP_DROP = 5,
    MOVE_BLOCK_ON_TARGET_2 = 6,
    #pickup
    MOVE_TARGET_1 = 7,
    # drop
    # move empty
    # pickup
    MOVE_TARGET_2 = 8
    # drop

ep_robot = robot.Robot()
ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YR")
ep_chassis = ep_robot.chassis
ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
#ep_arm = ep_robot.robotic_arm
#ep_arm.moveto(x=200, y=0).wait_for_completed()
ep_gripper = ep_robot.gripper

# Start printing the gripper position
# ep_arm.sub_position(freq=5, callback=sub_data_handler)

controller = IBVS_Controller(control_mode='4xyzy', interaction_mode='mean', num_pts=4)
controller.set_lambda_matrix([5.0, 10.0, 1.0, 1000.0]) # robot y velocity; robot z velocity; robot x velocity; robot z angular velocity
controller.set_desired_points([(-0.16, 0.125, 0.215), (0.16, 0.125, 0.215), (-0.16, 0.95, 0.215), (0.16, 0.95, 0.215)])

state = State.START_2B_2M
prev = State.START_2B_2M

while True:
    # ret, frame = vid.read()
    try:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
    except:
        frame = None
        continue
    if frame is not None:
        clean_frame = frame.copy()
        start = time.time()
        if model.predictor:
            model.predictor.args.verbose = False
        result = model.predict(source=frame, show=False)[0]
        
        boxes = result.boxes
        if len(boxes) == 0:
            # ep_chassis.drive_speed(x=0, y=0, z=-10, timeout=5)
            pass
        else:
            for box in boxes:
                if int(box.cls) != 0:
                    continue
                xyxy = box.xyxy.cpu().numpy().flatten()
                cv2.rectangle(frame,
                            (int(xyxy[0]), int(xyxy[1])), 
                            (int(xyxy[2]), int(xyxy[3])),
                            color=(0, 0, 255), thickness=2)
                
                corners = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                corners[0] = 2*((corners[0]) / 640) - 1
                corners[2] = 2*((corners[2]) / 640) - 1
                
                corners[1] = 2*((corners[1]) / 360) - 1
                corners[3] = 2*((corners[3]) / 360) - 1

                cv2.putText(frame, str((round(corners[2], 2), round(corners[1], 2))), (int(xyxy[2]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
                cv2.putText(frame, str((round(corners[0], 2), round(corners[3], 2))), (int(xyxy[0]), int(xyxy[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
                cv2.putText(frame, str((round(corners[2], 2), round(corners[3], 2))), (int(xyxy[2]), int(xyxy[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
                cv2.putText(frame, str((round(corners[0], 2), round(corners[1], 2))), (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
            
                detected_block = clean_frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                detected_block_gray = cv2.cvtColor(detected_block, cv2.COLOR_BGR2GRAY)
                detected_block_gray_gaussian = cv2.GaussianBlur(detected_block_gray, (3, 3), 0)
                detected_block_lines = cv2.Canny(detected_block_gray_gaussian, 100, 250, None, 3)

                # detected_block_lines_hough = cv2.HoughLines(detected_block_lines, 1, np.pi / 180, 30, None, 0, 0)
                detected_block_lines_hough = cv2.HoughLinesP(detected_block_lines, 1, np.pi / 180, 25, None, 15, 1)
                # print(detected_block_lines_hough) 
                
            
            if state == State.START_2B_2M:
                print("Start")
                
                # open gripper
                ep_gripper.open(power=100)
                time.sleep(2)
                ep_gripper.pause()
                
                state = State.MOVE_BLOCK_ON_TARGET_1
            elif state == State.MOVE_BLOCK_ON_TARGET_1:
                #print("State is START_2B_2M")
                        
                # print(corners)
                
                depth =(0.158*314.0)/(int(xyxy[3])-int(xyxy[1]))
                
                controller.set_current_points([(corners[0], corners[1], depth), (corners[2], corners[1], depth), (corners[0], corners[3], depth), (corners[2], corners[3], depth)])
                controller.calculate_interaction_matrix()
                vels = controller.calculate_velocities()
                # print(f"vels: {vels}")

                # robot x velocity is camera z velocity
                robot_x_velocity = vels[2][0]
                #print(robot_x_velocity)
                robot_x_velocity = clamp(robot_x_velocity, ROBOT_X_VELOCITY_MIN, ROBOT_X_VELOCITY_MAX)

                # robot y velocity is camera x velocity
                robot_y_velocity = vels[0][0]
                robot_y_velocity = clamp(robot_y_velocity, ROBOT_Y_VELOCITY_MIN, ROBOT_Y_VELOCITY_MAX)

                # robot z velocity is inverted camera y velocity
                robot_z_velocity = -vels[1][0]
                robot_z_velocity = clamp(robot_z_velocity, ROBOT_Z_VELOCITY_MIN, ROBOT_Z_VELOCITY_MAX)
                robot_z_position += robot_z_velocity
                robot_z_position = clamp(robot_z_position, ROBOT_Z_POSITION_MIN, ROBOT_Z_POSITION_MAX)

                # robot z angular velocity is camera y angular velocity
                robot_z_angular_velocity = vels[3][0]
                robot_z_angular_velocity = clamp(robot_z_angular_velocity, ROBOT_Z_ANGULAR_VELOCITY_MIN, ROBOT_Z_ANGULAR_VELOCITY_MAX)
                
                if detected_block_lines_hough is not None:
                    most_vertical = detected_block_lines_hough[0][0]
                    most_horizontal = detected_block_lines_hough[0][0]
                    for i in range(len(detected_block_lines_hough)):
                        # rho = detected_block_lines_hough[i][0][0]
                        # theta = detected_block_lines_hough[i][0][1]
                        # a = math.cos(theta)
                        # b = math.sin(theta)
                        # x0 = a * rho
                        # y0 = b * rho
                        # pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                        # pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                        # cv2.line(detected_block, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
                        l = detected_block_lines_hough[i][0]
                        if abs(l[2] - l[0]) < abs(most_vertical[2] - most_vertical[0]):
                            most_vertical = l
                        if abs(l[3] - l[1]) < abs(most_horizontal[3] - most_horizontal[1]):
                            most_horizontal = l
                    # cv2.line(detected_block, (most_vertical[0], most_vertical[1]), (most_vertical[2], most_vertical[3]), (0,0,0), 3, cv2.LINE_AA)
                    cv2.line(detected_block, (most_horizontal[0], most_horizontal[1]), (most_horizontal[2], most_horizontal[3]), (255,255,255), 3, cv2.LINE_AA)

                    if most_horizontal is not most_vertical:
                        # most_vertical_angle = math.atan2(most_vertical[3] - most_vertical[1], most_vertical[2] - most_vertical[0])
                        most_horizontal_angle = math.atan2(most_horizontal[3] - most_horizontal[1], most_horizontal[2] - most_horizontal[0])
                        # print(f"vertical {most_vertical_angle} horizontal {most_horizontal_angle}")
                        #print(f"horizontal {most_horizontal_angle}; rotation to align: {most_horizontal_angle}")
                    else:
                        most_horizontal_angle = 0
                else:
                    most_horizontal_angle = 0

                # send robot z position to arm
                #ep_arm.moveto(x=200, y=robot_z_position).wait_for_completed()

                # send robot x, y, and angular z velocities to robot
                #ep_chassis.drive_speed(x=robot_x_velocity, y=robot_y_velocity, z=robot_z_angular_velocity, timeout=5)
                #ep_chassis.drive_speed(x=robot_x_velocity, y=robot_y_velocity, z=10.0*most_horizontal_angle, timeout=5)

                
                controller.calculate_error_vector()
                err_nrm = np.linalg.norm(controller.errs)
                #if depth < 0.22:
                #    state = State.GRIP_PICKUP
                #    prev = State.MOVE_BLOCK_ON_TARGET_1

                print(f"depth: {depth} err_nrm: {err_nrm} vels: x {robot_x_velocity} y {robot_y_velocity} z {robot_z_velocity} z ang {robot_z_angular_velocity}; arm pos: {robot_z_position}")
                        
            elif state == State.GRIP_PICKUP:
                
                
                ep_gripper.close(power=150)
                time.sleep(0.5)
                ep_gripper.pause()
                
                ep_arm.move(x=0, y=30).wait_for_completed()
                
                state = State.MOVE_EMPTY_1
                prev = State.GRIP_PICKUP
                
            elif state == State.MOVE_EMPTY_1:
            
                
                print("State is MOVE_EMPTY_1")
            elif state == State.GRIP_DROP:
                print("State is GRIP_DROP")
            elif state == State.MOVE_BLOCK_ON_TARGET_2:
                print("State is MOVE_BLOCK_ON_TARGET_2")
            elif state == State.MOVE_TARGET_1:
                print("State is MOVE_TARGET_1")
            elif state == State.MOVE_TARGET_2:
                print("State is MOVE_TARGET_2")
            else:
                print("Unknown state")

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
ep_robot.close()