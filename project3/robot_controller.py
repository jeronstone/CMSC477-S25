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

class Robot():
    def __init__(self, ep_robot):
        # robomaster variables
        self.ep_robot = ep_robot
        self.ep_chassis = ep_robot.chassis
        self.ep_chassis.sub_attitude(freq=5, callback=self.attitude_callback)
        self.ep_chassis.sub_position(cs=0, freq=5, callback=self.chassis_callback)

        # vision controller
        self.vision = Vision(r"C:\Users\jesto\Desktop\CMSC477\CMSC477-S25\runs\detect\train2\weights\best.pt")

        # minimax agent
        self.minimax_agent = MiniMaxAgent(State(), 2*2)

        # map controller
        self.map = MapController(ep_robot)

        # IBVS controller
        self.controller = IBVS_Controller(control_mode='2xz', interaction_mode='mean', num_pts=4)
        self.controller.set_lambda_matrix([3.0, 1.25]) # robot y velocity; robot x velocity
        self.controller.set_desired_points([(-0.2, -0.7, 0.18), (0.2, -0.7, 0.18), (-0.2, 1.0, 0.18), (0.2, 1.0, 0.18)])
        
        self.our_position = (1.0,1.0)

    def chassis_callback(self, pos):
        x, y, z = pos
        # print(f"x: {x} y: {y} z: {z}")
        self.our_position = (float(x)+1.0, -float(y-1.0))
        print(f"current position: {self.our_position}")

    def attitude_callback(self, pos):
        yaw, pitch, roll = pos
        print(f"yaw: {yaw} pitch: {pitch} roll: {roll}")
       
    def grip_pickup(self):
            
        self.ep_chassis.move(x=0.05, y=0, z=0, xy_speed=0.2).wait_for_completed(2.0) # move slightly forward to position tower in gripper
        time.sleep(2.0)
        
        self.ep_gripper.close(power=125)
        time.sleep(1.0)
        self.ep_gripper.pause()
        
        self.ep_arm.moveto(x=200, y=-10).wait_for_completed(2.0)
        time.sleep(2.0)
        
    def grip_drop(self):

        self.ep_arm.moveto(x=200, y=-50).wait_for_completed(2.0)
        time.sleep(2.0)
        
        self.ep_gripper.open(power=50)
        time.sleep(1.0)
        self.ep_gripper.pause()
        
        time.sleep(2.0)
        
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
    
    
    DIST_THRESH_X = 0.1
    DIST_THRESH_Y = 0.1
    '''
    Moves to global position x, y on the map using simple p loop and constant speed
    '''
    def move_to_xy(self, desired_x, desired_y):
        
        err_x = self.our_position[0] - desired_x
        err_y = self.our_position[1] - desired_y

        while abs(err_x) > self.DIST_THRESH_X or abs(err_y) > self.DIST_THRESH_Y:
            
            err_x = self.our_position[0] - desired_x
            err_y = self.our_position[1] - desired_y
            
            print(f'Error: {err_x} {err_y}')
            
            velo_x = 0.0
            velo_y = 0.0
            
            if abs(err_x) > self.DIST_THRESH_X:
                velo_x = -math.copysign(2.0, err_x)
                
            if abs(err_y) > self.DIST_THRESH_Y:
                velo_y = -math.copysign(2.0, err_y)
            
            self.ep_chassis.drive_speed(x=velo_x, y=velo_y, z=0.0, timeout=5)
            time.sleep(0.1)
            
        self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
        return 1
            
            
if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH7T001008H")
    _robot = Robot(ep_robot)
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    ep_led = ep_robot.led
    
    time.sleep(1.0)
    print('5 sec pass')
    
    _robot.move_to_xy(1.5,1.5)
    
    print('done')
    exit(0)