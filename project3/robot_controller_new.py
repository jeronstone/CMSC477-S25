### temp file ###
from util import *
from state import *
from vision import *
from map_controller import *
from ibvs_controller import *
from state import *
from agents import *

import cv2
from robomaster import robot
from robomaster import camera
from queue import Empty
import time
import math

ROBOT_X_VELOCITY_MIN = -0.2
ROBOT_X_VELOCITY_MAX = 0.2
ROBOT_Y_VELOCITY_MIN = -0.2
ROBOT_Y_VELOCITY_MAX = 0.2
ROBOT_Z_VELOCITY_MIN = -1.0
ROBOT_Z_VELOCITY_MAX = 1.0
ROBOT_Z_POSITION_MIN = -50
ROBOT_Z_POSITION_MAX = -10
ROBOT_Z_ANGULAR_VELOCITY_MIN = -0.5
ROBOT_Z_ANGULAR_VELOCITY_MAX = 0.5

DIST_THRESH_X = 0.1
DIST_THRESH_Y = 0.1

class Robot():
    def __init__(self, ep_robot):
        # robomaster variables
        self.ep_robot = ep_robot
        self.ep_chassis = ep_robot.chassis
        self.ep_chassis.sub_attitude(freq=10, callback=self.attitude_callback)
        self.ep_chassis.sub_position(cs=0, freq=10, callback=self.chassis_callback)
        self.ep_gripper = ep_robot.gripper
        self.ep_arm = ep_robot.robotic_arm
        self.ep_led = ep_robot.led

        # position/rotation variables
        self.our_position = (1.0, 1.0)
        self.our_rotation = None
        self.frame_rotation = None
        
        self.curr_state = "MOVE_CLOSET"#"MOVE_LEFTMOST_BLOCK"

        # vision controller
        self.vision = Vision(r"..\runs\detect\train2\weights\best.pt")

        # minimax agent
        self.minimax_agent = MiniMaxAgent(State(), 2*2)

        # map controller
        #self.map = MapController(ep_robot)

        # IBVS controller
        self.controller = IBVS_Controller(control_mode='2xz', interaction_mode='mean', num_pts=4)
        self.controller.set_lambda_matrix([3.0, 1.25]) # robot y velocity; robot x velocity
        self.controller.set_desired_points([(-0.16, 0.375, 0.18), (0.16, 0.375, 0.18), (-0.16, 0.95, 0.18), (0.16, 0.95, 0.18)])

    def chassis_callback(self, pos):
        x, y, z = pos
        # print(f"x: {x} y: {y} z: {z}")
        # self.our_position = (float(x)+1.0, -float(y-1.0))
        #self.our_position = (float(y), float(x))
        if self.frame_rotation is None: # if we don't have a frame rotation yet, just assume we haven't moved
            self.our_position = (3.0/FEET_TO_METER_DIV_BY, 3.0/FEET_TO_METER_DIV_BY)
        else:
            rotated_xy = self.frame_rotation @ np.array([[float(x)], [float(y)]])
            self.our_position = (rotated_xy[1][0] + 3.0/FEET_TO_METER_DIV_BY, rotated_xy[0][0] + 3.0/FEET_TO_METER_DIV_BY)
        
        #print(f"current position: {self.our_position}")
        #print(self.get_current_location())

    def attitude_callback(self, pos):
        yaw, pitch, roll = pos
        if self.frame_rotation is None: # the first time we read the attitude, our yaw should be +90 in the global robot frame (which means it should point in +x in our frame). Create the rotation matrix from this initial angle reading
            theta = np.radians(yaw - 90) # how offset we are from +90
            c, s = np.cos(-theta), np.sin(-theta) # we want the rotation matrix to be the opposite of that angle
            self.frame_rotation = np.array([[c, -s], [s, c]]) # final rotation matrix
            #print(self.frame_rotation)
        else:
            theta = np.radians(yaw)
            self.our_rotation = self.frame_rotation @ np.array([[np.cos(theta)], [np.sin(theta)]])
            curr_theta = np.arctan2(self.our_rotation[1], self.our_rotation[0])
            #print(f"current rotation: {curr_theta}")

    def grip_pickup(self):
            
        self.ep_chassis.move(x=0.05, y=0, z=0, xy_speed=0.2).wait_for_completed(2.0) # move slightly forward to position tower in gripper
        time.sleep(2.0)
        
        self.ep_gripper.close(power=125)
        time.sleep(1.0)
        self.ep_gripper.pause()
        
        self.ep_arm.moveto(x=200, y=-10).wait_for_completed(2.0)
        time.sleep(2.0)
        
        self.curr_state = "DONE"
        
    def grip_drop(self):

        self.ep_arm.moveto(x=200, y=-50).wait_for_completed(2.0)
        time.sleep(2.0)
        
        self.ep_gripper.open(power=50)
        time.sleep(1.0)
        self.ep_gripper.pause()
        
        time.sleep(2.0)

    def move_to_leftmost_block(self, frame):
            
        fr, detections = self.vision.get_yolo_pred(frame, hough=True)
        
        if len(detections) == 0:
            self.ep_chassis.drive_speed(x=0, y=0, z=20, timeout=5)
            return fr, 0
        else:
        
            # TODO determine which detection to follow
            # blocks are cls = 2, 3, 4
            
            leftmost = 999
            leftmost_idx = -1
            for i, d in enumerate(detections):
                cls, corners, depth, detected_block_lines_hough = d
                if cls >= 2 and cls <= 4: # if its a block
                    if corners[0] < leftmost: # if its more left than a previous block
                        leftmost = corners[0]
                        leftmost_idx = i
                        
            if leftmost_idx == -1:
                print(f'No leftmost block detected')
                self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
                return fr, -1
            
            cls, corners, depth, detected_block_lines_hough = detections[leftmost_idx]
                    
            self.controller.set_current_points([(corners[0], corners[1], depth), (corners[2], corners[1], depth), (corners[0], corners[3], depth), (corners[2], corners[3], depth)])
            self.controller.calculate_interaction_matrix()
            vels = self.controller.calculate_velocities()

            # robot x velocity is camera z velocity
            robot_x_velocity = vels[1][0]
            robot_x_velocity = clamp(robot_x_velocity, ROBOT_X_VELOCITY_MIN, ROBOT_X_VELOCITY_MAX)

            # robot y velocity is camera x velocity
            robot_y_velocity = vels[0][0]
            robot_y_velocity = clamp(robot_y_velocity, ROBOT_Y_VELOCITY_MIN, ROBOT_Y_VELOCITY_MAX)

            # hough transform for rotational velocity
            most_horizontal_angle = 0
            if detected_block_lines_hough is not None:
                most_vertical = detected_block_lines_hough[0][0]
                most_horizontal = detected_block_lines_hough[0][0]
                for i in range(len(detected_block_lines_hough)):
                    l = detected_block_lines_hough[i][0]
                    if abs(l[2] - l[0]) < abs(most_vertical[2] - most_vertical[0]):
                        most_vertical = l
                    if abs(l[3] - l[1]) < abs(most_horizontal[3] - most_horizontal[1]):
                        most_horizontal = l
                #cv2.line(detected_block, (most_vertical[0], most_vertical[1]), (most_vertical[2], most_vertical[3]), (0,0,0), 3, cv2.LINE_AA)
                #cv2.line(detected_block, (most_horizontal[0], most_horizontal[1]), (most_horizontal[2], most_horizontal[3]), (255,255,255), 3, cv2.LINE_AA)
                #cv2.imshow('detected_block', detected_block)
                #key = cv2.waitKey(1)
                #cv2.imshow('detected block', detected_block)

                if most_horizontal is not most_vertical:
                    # most_vertical_angle = math.atan2(most_vertical[3] - most_vertical[1], most_vertical[2] - most_vertical[0])
                    most_horizontal_angle = math.atan2(most_horizontal[3] - most_horizontal[1], most_horizontal[2] - most_horizontal[0])
                    # print(f"vertical {most_vertical_angle} horizontal {most_horizontal_angle}")
                    #print(f"horizontal {most_horizontal_angle}; rotation to align: {most_horizontal_angle}")


            # send robot x, y, and angular z velocities to robot
            #ep_chassis.drive_speed(x=robot_x_velocity, y=robot_y_velocity, z=robot_z_angular_velocity, timeout=5)
            self.ep_chassis.drive_speed(x=robot_x_velocity, y=robot_y_velocity, z=10.0*most_horizontal_angle, timeout=5)
            
            self.controller.calculate_error_vector()
            err_nrm = np.linalg.norm(self.controller.errs)
            if depth < 0.19 and err_nrm < 0.16 and abs(most_horizontal_angle) < 0.05: # within 20 cm of camera, errors in point positions less than 0.125 normalized image distance, and most horizontal angle in block is within 0.05 radians
            #if corners[1] > 0.06 and corners[3] > 0.95 and corners[0] > -0.2 and corners[2] < 0.2:
                print('close to block, transition')
                self.curr_state = "GRIP_PICKUP"
                return fr, 1
            print(f"horiz_ang: {most_horizontal_angle} depth: {depth} err_nrm: {err_nrm} vels: x {robot_x_velocity} y {robot_y_velocity}")
            return fr, 0
        
    def move_to_block(self, frame):
        
        fr, detections = self.vision.get_yolo_pred(frame, hough=True)
        
        if len(detections) == 0:
            self.ep_chassis.drive_speed(x=0, y=0, z=20, timeout=5)
        else:
        
            # TODO determine which detection to follow
            # blocks are cls = 2, 3, 4
            cls, corners, detected_block_lines_hough, depth = detections[0]
                    
            controller.set_current_points([(corners[0], corners[1], depth), (corners[2], corners[1], depth), (corners[0], corners[3], depth), (corners[2], corners[3], depth)])
            controller.calculate_interaction_matrix()
            vels = controller.calculate_velocities()

            # robot x velocity is camera z velocity
            robot_x_velocity = vels[1][0]
            robot_x_velocity = clamp(robot_x_velocity, ROBOT_X_VELOCITY_MIN, ROBOT_X_VELOCITY_MAX)

            # robot y velocity is camera x velocity
            robot_y_velocity = vels[0][0]
            robot_y_velocity = clamp(robot_y_velocity, ROBOT_Y_VELOCITY_MIN, ROBOT_Y_VELOCITY_MAX)

            # hough transform for rotational velocity
            most_horizontal_angle = 0
            if detected_block_lines_hough is not None:
                most_vertical = detected_block_lines_hough[0][0]
                most_horizontal = detected_block_lines_hough[0][0]
                for i in range(len(detected_block_lines_hough)):
                    l = detected_block_lines_hough[i][0]
                    if abs(l[2] - l[0]) < abs(most_vertical[2] - most_vertical[0]):
                        most_vertical = l
                    if abs(l[3] - l[1]) < abs(most_horizontal[3] - most_horizontal[1]):
                        most_horizontal = l
                #cv2.line(detected_block, (most_vertical[0], most_vertical[1]), (most_vertical[2], most_vertical[3]), (0,0,0), 3, cv2.LINE_AA)
                #cv2.line(detected_block, (most_horizontal[0], most_horizontal[1]), (most_horizontal[2], most_horizontal[3]), (255,255,255), 3, cv2.LINE_AA)
                #cv2.imshow('detected_block', detected_block)
                #key = cv2.waitKey(1)
                #cv2.imshow('detected block', detected_block)

                if most_horizontal is not most_vertical:
                    # most_vertical_angle = math.atan2(most_vertical[3] - most_vertical[1], most_vertical[2] - most_vertical[0])
                    most_horizontal_angle = math.atan2(most_horizontal[3] - most_horizontal[1], most_horizontal[2] - most_horizontal[0])
                    # print(f"vertical {most_vertical_angle} horizontal {most_horizontal_angle}")
                    #print(f"horizontal {most_horizontal_angle}; rotation to align: {most_horizontal_angle}")


            # send robot x, y, and angular z velocities to robot
            #ep_chassis.drive_speed(x=robot_x_velocity, y=robot_y_velocity, z=robot_z_angular_velocity, timeout=5)
            self.ep_chassis.drive_speed(x=robot_x_velocity, y=robot_y_velocity, z=10.0*most_horizontal_angle, timeout=5)
            
            controller.calculate_error_vector()
            err_nrm = np.linalg.norm(controller.errs)
            if depth < 0.19 and err_nrm < 0.16 and abs(most_horizontal_angle) < 0.05: # within 20 cm of camera, errors in point positions less than 0.125 normalized image distance, and most horizontal angle in block is within 0.05 radians
            #if corners[1] > 0.06 and corners[3] > 0.95 and corners[0] > -0.2 and corners[2] < 0.2:
                print('close to block, transition')
            print(f"horiz_ang: {most_horizontal_angle} depth: {depth} err_nrm: {err_nrm} vels: x {robot_x_velocity} y {robot_y_velocity}")
    
    '''
    Moves to global position x, y on the map using simple p loop and constant speed
    '''
    def move_to_xy(self, desired_x, desired_y):
        
        err_x = self.our_position[0] - desired_x
        err_y = self.our_position[1] - desired_y

        if abs(err_x) > DIST_THRESH_X or abs(err_y) > DIST_THRESH_Y:
            
            err_x = self.our_position[0] - desired_x
            err_y = self.our_position[1] - desired_y
            
            velo_x = 0.0
            velo_y = 0.0
            
            if abs(err_x) > DIST_THRESH_X:
                velo_x = -math.copysign(0.75, err_x)
                
            if abs(err_y) > DIST_THRESH_Y:
                velo_y = math.copysign(0.75, err_y)
                
            print(f'Error: {err_x} {err_y} \t Velos: {velo_x} {velo_y}')
            
            self.ep_chassis.drive_speed(x=velo_x, y=velo_y, z=0.0, timeout=5)
            time.sleep(0.1)
            
            return 0
        else:
            self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
            self.curr_state = "DONE"
            return 1
    
if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH7T001008H")
    _robot = Robot(ep_robot)
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    ep_led = ep_robot.led
    
    x_vel = 0.0
    y_vel = 0.0
    z_vel = 0.0
    
    state_done_flag = False
        
    while True:

        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue

        # if frame is not None:
        #     start = time.time()
        #     if model.predictor:
        #         model.predictor.args.verbose = False
        #     result = model.predict(source=frame, show=False)[0]


        #     # DIY visualization is much faster than show=True for some reason
        #     boxes = result.boxes
        #     for box in boxes:
        #         xyxy = box.xyxy.cpu().numpy().flatten()
        #         cls = int(box.cls)
        #         class_label = result.names[cls]
        #         cv2.rectangle(frame,
        #                     (int(xyxy[0]), int(xyxy[1])), 
        #                     (int(xyxy[2]), int(xyxy[3])),
        #                     color=(0, 0, 255), thickness=2)
                
        #         cv2.putText(frame, class_label, (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        
        key = cv2.waitKey(1)
        if key == ord('w'):
            x_vel += 0.1
        elif key == ord('s'):
            x_vel -= 0.1
        elif key == ord('a'):
            y_vel -= 0.1
        elif key == ord('d'):
            y_vel += 0.1
        elif key == ord('q'):
            z_vel -= 5.0
        elif key == ord('e'):
            z_vel += 5.0
        elif key == ord(' '):
            x_vel = 0.0
            y_vel = 0.0
            z_vel = 0.0
        elif key == ord('z'):
            break
        
        led_red = clamp(int(abs(x_vel) * 128), 0, 255)
        led_green = clamp(int(abs(y_vel) * 128), 0, 255)
        led_blue = clamp(int(abs(z_vel) * 10), 0, 255)
        # _robot.ep_chassis.drive_speed(x=x_vel, y=y_vel, z=z_vel, timeout=5)
        ep_led.set_led(comp='all', r=led_red, g=led_green, b=led_blue, effect='on')
        
        if state_done_flag:
            # todo get new action
            print("done flag true")
            break
            state_done_flag = False
        
        
        fr = None
        if _robot.curr_state == "MOVE_LEFTMOST_BLOCK":
            fr, ret = _robot.move_to_leftmost_block(frame)
        elif _robot.curr_state == "GRIP_PICKUP":
            _robot.grip_pickup()
        elif _robot.curr_state == "MOVE_CLOSET":
            _robot.move_to_xy(OUR_CLOSET_PICKUP[0], OUR_CLOSET_PICKUP[1])
        elif _robot.curr_state == "DONE":
            state_done_flag = True
        
        
        # fr, ret = _robot.move_to_leftmost_block(frame)
        # if ret == 1:
        #     break

        if fr is not None:
            cv2.imshow("img", fr)

    _robot.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
    ep_robot.close()
    exit(0)