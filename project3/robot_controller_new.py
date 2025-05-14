### temp file ###
from util import *
from state import *
from vision import *
from map_controller import *
from ibvs_controller import *
from state import *
from agents import *
from ApriltagDetector import *

import matplotlib.pyplot as plt
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
APRILTAG_CLOSE_TRESH = 0.25
ROBOT_CLOSE_THRESH = 0.2

FEET_TO_METER_DIV_BY = 3.281

ROBOT_SIZE_EST = 0.25 #23 cm width, give it +2cm if its kinda sideways

# location boundaries
OUR_CLOSET_BOUNDARY = [(10.0/FEET_TO_METER_DIV_BY, 2.0/FEET_TO_METER_DIV_BY), (11.75/FEET_TO_METER_DIV_BY, 5.0/FEET_TO_METER_DIV_BY)]
OUR_ROOM_BOUNDARY = [(0.25/FEET_TO_METER_DIV_BY, 0.25/FEET_TO_METER_DIV_BY), (10.25/FEET_TO_METER_DIV_BY, 9.25/FEET_TO_METER_DIV_BY)]
HALLWAY_BOUNDARY = [(4.5/FEET_TO_METER_DIV_BY, 9.25/FEET_TO_METER_DIV_BY), (7.5/FEET_TO_METER_DIV_BY, 11.75/FEET_TO_METER_DIV_BY)]
THEIR_ROOM_BOUNDARY = [(1.75/FEET_TO_METER_DIV_BY, 11.75/FEET_TO_METER_DIV_BY), (11.75/FEET_TO_METER_DIV_BY, 20.75/FEET_TO_METER_DIV_BY)]
THEIR_CLOSET_BOUNDARY = [(0.25/FEET_TO_METER_DIV_BY, 16.0/FEET_TO_METER_DIV_BY), (2.0/FEET_TO_METER_DIV_BY, 19.0/FEET_TO_METER_DIV_BY)]

# pickup/dropoff locations
OUR_CLOSET_PICKUP = (2.72, 1.025)
OUR_CLOSET_DROPOFF = (3.0, 0.75) # increment y by 0.2 each time we drop off
OUR_ROOM_MOVE = (2.0, 1.025)
OUR_ROOM_PICKUP = (2.585, 1.110)
OUR_ROOM_DROPOFF = (1.5, 2.20) # decrement y by 0.2 each time we drop off
HALLWAY_MOVE = (2.0, 3.05)
THEIR_ROOM_MOVE = (2.0, 5.426)
THEIR_ROOM_PICKUP = (2.142, 5.426)
THEIR_ROOM_DROPOFF = (2.9, 4.1) # increment y by 0.2 each time we drop off
THEIR_CLOSET_PICKUP = (2.142, 5.426)
THEIR_CLOSET_DROPOFF = (1.512, 5.375) # increment y by 0.2 each time we drop off

position_history_x = []
position_history_y = []

class Robot():
    def __init__(self, ep_robot):
        # robomaster variables
        self.ep_robot = ep_robot
        self.ep_chassis = ep_robot.chassis
        self.ep_chassis.sub_attitude(freq=10, callback=self.attitude_callback)
        self.ep_chassis.sub_position(cs=0, freq=10, callback=self.chassis_callback)
        ep_robot.sensor.sub_distance(freq=10, callback=self.dist_callback)
        self.curr_ir_dist = 65534
        self.ep_gripper = ep_robot.gripper
        self.ep_arm = ep_robot.robotic_arm
        self.ep_led = ep_robot.led

        # position/rotation variables
        self.our_position = (3.0/FEET_TO_METER_DIV_BY, 3.0/FEET_TO_METER_DIV_BY)
        self.our_rotation = None
        self.curr_theta = 0.0
        self.frame_rotation = None

        # vision controller
        self.vision = Vision(r"..\runs\detect\train2\weights\best.pt")
        K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel
        marker_size_m = 0.153 # Size of the AprilTag in meters
        self.apriltag_detector = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)

        # minimax agent
        self.minimax_agent = MiniMaxAgent(State(), 2*2)
        self.curr_action = None
        self.curr_state = "DONE"#"MOVE_LEFTMOST_BLOCK"
        self.prev_state  = "DONE"

        # map controller
        #self.map = MapController(ep_robot)

        # IBVS controller
        self.controller = IBVS_Controller(control_mode='2xz', interaction_mode='mean', num_pts=4)
        self.controller.set_lambda_matrix([3.0, 1.0]) # robot y velocity; robot x velocity
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
            self.our_position = (rotated_xy[0][0] + 3.0/FEET_TO_METER_DIV_BY, -rotated_xy[1][0] + 3.0/FEET_TO_METER_DIV_BY)
            position_history_x.append(self.our_position[0])
            position_history_y.append(self.our_position[1])
        
        #print(f"current position: {self.our_position}")
        #print(self.get_current_location())
    
    def dist_callback(self, dist):
        self.curr_ir_dist = dist
        
    def set_frame_rotation(self, desired_heading):
        theta = np.radians(self.prev_yaw - desired_heading) # how offset we are from +90
        c, s = np.cos(-theta), np.sin(-theta) # we want the rotation matrix to be the opposite of that angle
        self.frame_rotation = np.array([[c, -s], [s, c]]) # final rotation matrix

    def attitude_callback(self, pos):
        yaw, pitch, roll = pos
        self.prev_yaw = yaw
        if self.frame_rotation is None: # the first time we read the attitude, our yaw should be +90 in the global robot frame (which means it should point in +x in our frame). Create the rotation matrix from this initial angle reading
            self.set_frame_rotation(90)
            #print(self.frame_rotation)
        else:
            theta = np.radians(yaw)
            self.our_rotation = self.frame_rotation @ np.array([[np.cos(theta)], [np.sin(theta)]])
            self.curr_theta = np.arctan2(self.our_rotation[1], self.our_rotation[0])
            #print(f"current rotation: {curr_theta}")

    def get_current_location(self):
        if OUR_CLOSET_BOUNDARY[0][0] <= self.our_position[0] <= OUR_CLOSET_BOUNDARY[1][0] and OUR_CLOSET_BOUNDARY[0][1] <= self.our_position[1] <= OUR_CLOSET_BOUNDARY[1][1]:
            return "OUR_CLOSET"
        elif OUR_ROOM_BOUNDARY[0][0] <= self.our_position[0] <= OUR_ROOM_BOUNDARY[1][0] and OUR_ROOM_BOUNDARY[0][1] <= self.our_position[1] <= OUR_ROOM_BOUNDARY[1][1]:
            return "OUR_ROOM"
        elif HALLWAY_BOUNDARY[0][0] <= self.our_position[0] <= HALLWAY_BOUNDARY[1][0] and HALLWAY_BOUNDARY[0][1] <= self.our_position[1] <= HALLWAY_BOUNDARY[1][1]:
            return "HALLWAY"
        elif THEIR_ROOM_BOUNDARY[0][0] <= self.our_position[0] <= THEIR_ROOM_BOUNDARY[1][0] and THEIR_ROOM_BOUNDARY[0][1] <= self.our_position[1] <= THEIR_ROOM_BOUNDARY[1][1]:
            return "THEIR_ROOM"
        elif THEIR_CLOSET_BOUNDARY[0][0] <= self.our_position[0] <= THEIR_CLOSET_BOUNDARY[1][0] and THEIR_CLOSET_BOUNDARY[0][1] <= self.our_position[1] <= THEIR_CLOSET_BOUNDARY[1][1]:
            return "THEIR_CLOSET"
        else:
            return "undef"
        
    def update_state_with_detections(self, location):

        fr, detections = self.vision.get_yolo_pred(frame, hough=False)

        if len(detections) > 0:
            block_list = [block for block in self.minimax_agent.curr_state.block_positions if block[1] != location]
            for detection in detections:
                cls, _, _, _ = detection
                if cls == 0: # enemy robot
                    self.minimax_agent.curr_state.their_position = location
                elif 2 <= cls <= 4: # block
                    block_list.append(("4x4", location))
            self.minimax_agent.curr_state.block_positions = block_list

        self.curr_state = "DONE"
        return fr, 0

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
        
        self.ep_gripper.open(power=125)
        time.sleep(1.0)
        self.ep_gripper.pause()
        
        time.sleep(2.0)

        self.curr_state = "DONE"

    def move_to_leftmost_block(self, frame):
            
        fr, detections = self.vision.get_yolo_pred(frame, hough=True)
        
        if len(detections) == 0:
            self.ep_chassis.drive_speed(x=0, y=0, z=20, timeout=5)
            return fr, -1
        else:            
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
                self.prev_state = self.curr_state
                self.curr_state = "GRIP_PICKUP"
                return fr, 1
            print(f"horiz_ang: {most_horizontal_angle} depth: {depth} err_nrm: {err_nrm} vels: x {robot_x_velocity} y {robot_y_velocity}")
            return fr, 0
        
    def move_to_block(self, frame):
        
        fr, detections = self.vision.get_yolo_pred(frame, hough=True)
        
        if len(detections) == 0:
            self.ep_chassis.drive_speed(x=0, y=0, z=20, timeout=5)
        else:
        
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
    def move_to_xy(self, desired_x, desired_y, desired_heading, final_location, avoid_obstacles=False, frame=None):
        
        if avoid_obstacles and frame is None:
            # could get rid of the bool and just None check to determine if detecting 
            # but I like the explicivity here just to be safe
            return None, -1
        
        err_x_w = self.our_position[0] - desired_x # x error in world frame
        err_y_w = self.our_position[1] - desired_y # y error in world frame

        if abs(err_x_w) > DIST_THRESH_X or abs(err_y_w) > DIST_THRESH_Y:
            
            velo_x_w = -0.5 * err_x_w # x vel in world frame
            velo_y_w = 0.5 * err_y_w # y vel in world frame
            
            # if abs(err_x_w) > DIST_THRESH_X:
            #     velo_x_w = -math.copysign(0.4, err_x_w)
                
            # if abs(err_y_w) > DIST_THRESH_Y:
            #     velo_y_w = math.copysign(0.4, err_y_w)
                
            print(f'Error: {err_x_w} {err_y_w} \t Velos: {velo_x_w} {velo_y_w}')

            velos_r = self.frame_rotation @ np.array([[float(velo_x_w)], [float(velo_y_w)]])

            velo_x_r = clamp(velos_r.item(0), ROBOT_X_VELOCITY_MIN, ROBOT_X_VELOCITY_MAX)
            velo_y_r = clamp(velos_r.item(1), ROBOT_Y_VELOCITY_MIN, ROBOT_Y_VELOCITY_MAX)
            
            self.ep_chassis.drive_speed(x=velo_x_r, y=velo_y_r, z=0.0, timeout=5)
            #time.sleep(0.1)
            
            if avoid_obstacles:
                #print("detecing obstacles to avoid...")
                
                if self.curr_ir_dist < 150: #TODO change?
                    self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
                    self.prev_state = self.curr_state
                    self.curr_state = "AVOID_OBSTACLE_IR_FALLBACk"
                    return None, 0
                    
                    
                fr, detections = self.vision.get_yolo_pred(frame, hough=False, depth_len=ROBOT_SIZE_EST)
                for i, d in enumerate(detections):
                    cls, corners, depth, detected_block_lines_hough = d
                    if cls == 0: # robot detected
                        intheway = (depth > ROBOT_CLOSE_THRESH)
                        if intheway:
                            self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
                            self.prev_state = self.curr_state
                            self.curr_state = "AVOID_OBSTACLE_ROBOT"
                            return fr, 0
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray.astype(np.uint8)

                detections = self.apriltag_detector.find_tags(gray)
                print(f"apriltag detector detected: {len(detections)} apriltags")
                for detection in detections:
                    t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
                    distance = np.linalg.norm(t_ca-np.array([0, 0, APRILTAG_SIZE]))
                    print(f'Apriltag dist: {distance}')
                    if distance < APRILTAG_CLOSE_TRESH:
                        self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
                        self.prev_state = self.curr_state
                        self.curr_state = "AVOID_OBSTACLE_APRILTAG"
                        return fr, 0
            
            return None, 1
        else:
            self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
            time.sleep(0.1)
            self.ep_chassis.move(x=0, y=0, z=int(np.rad2deg(self.curr_theta) - desired_heading), z_speed=45).wait_for_completed(2.0)
            time.sleep(2.0)
            self.set_frame_rotation(desired_heading)
            self.minimax_agent.curr_state.our_position = final_location
            self.prev_state = self.curr_state
            self.curr_state = "UPDATE_STATE"
            return None, 0
        
    def avoid_obstacle(self, object, frame):
        print(f'Avoiding {object}')
        if object == "APRILTAG":
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray.astype(np.uint8)

            detections = self.apriltag_detector.find_tags(gray)
            
            if len(detections) != 0:
                for detection in detections:
                    t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
                    distance = np.linalg.norm(t_ca-np.array([0, 0, APRILTAG_SIZE]))
                    print(f'Apriltag dist: {distance}')
                    if distance < APRILTAG_CLOSE_TRESH:
                        print("There's an Apriltag thats too close still")
                        
                        pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)
                        top_left = tuple(pts[0][0])  # First corner
                        # top_right = tuple(pts[1][0])  # Second corner
                        # bottom_right = tuple(pts[2][0])  # Third corner
                        # bottom_left = tuple(pts[3][0])  # Fourth corner
                        if top_left > 0:    # right side, move left
                            self.ep_chassis.drive_speed(x=0.0, y=-0.5, z=0.0, timeout=5)
                        else:               # left side, move right
                            self.ep_chassis.drive_speed(x=0.0, y=0.5, z=0.0, timeout=5)
                        
                        return 1
            
            print("Obstacle avoided")
            # at this point, all detections were greater than thresh, or there were 0 detections
            self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
            self.curr_state = self.prev_state
            return 0
                
        elif object == "ROBOT":
            self.ep_chassis.drive_speed(x=0.0, y=-0.5, z=0.0, timeout=5)
            fr, detections = self.vision.get_yolo_pred(frame, hough=False, depth_len=ROBOT_SIZE_EST)
            for i, d in enumerate(detections):
                cls, corners, depth, detected_block_lines_hough = d
                if cls == 0: # robot detected
                    intheway = (depth > ROBOT_CLOSE_THRESH)
                    if intheway:
                        print("Theres a robot thats too close still")
                        
                        if corners[0] > 0:  # right side, move left
                            self.ep_chassis.drive_speed(x=0.0, y=-0.5, z=0.0, timeout=5)
                        else:               # left side, move right
                            self.ep_chassis.drive_speed(x=0.0, y=0.5, z=0.0, timeout=5)
                        
                        return 1
            
            print("Obstacle avoided")
            self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
            self.curr_state = self.prev_state
            return 0
        elif object == "IR_SENSOR":
            if self.curr_ir_dist < 200: #TODO thresh
                # drive backwards slowly
                self.ep_chassis.drive_speed(x=-0.3, y=0.0, z=0.0, timeout=5)
                return 1
            else:
                print("Obstacle avoided")
                self.curr_state = self.prev_state
                self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
                return 0
    
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
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 7)
    
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
        
        # update state based on observations
        
        if frame is None:
            time.sleep(0.1)
            continue
        

        if state_done_flag:
            print("done flag true")
            state_done_flag = False
            _robot.minimax_agent.perform_action(_robot.curr_action)
            _robot.curr_action = _robot.minimax_agent.choose_action()
            if _robot.curr_action == Action.PICKUP_BLOCK_2x2 or _robot.curr_action == Action.PICKUP_BLOCK_2x4 or _robot.curr_action == Action.PICKUP_BLOCK_4x4:
                _robot.curr_state = "MOVE_LEFTMOST_BLOCK"
            elif _robot.curr_action == Action.DROP_BLOCK:
                _robot.curr_state = "GRIP_DROP"
            elif _robot.curr_action == Action.MOVE_OUR_CLOSET:
                _robot.curr_state = "MOVE_OUR_CLOSET"
            elif _robot.curr_action == Action.MOVE_OUR_ROOM:
                _robot.curr_state = "MOVE_OUR_ROOM"
            elif _robot.curr_action == Action.MOVE_HALLWAY:
                _robot.curr_state = "MOVE_HALLWAY"
            elif _robot.curr_action == Action.MOVE_THEIR_ROOM:
                _robot.curr_state = "MOVE_THEIR_ROOM"
            elif _robot.curr_action == Action.MOVE_THEIR_CLOSET:
                _robot.curr_state = "MOVE_THEIR_CLOSET"
            else:
                _robot.curr_state = "DONE"
        
        print(f"curr state: {_robot.curr_state}")
        fr = None
        if _robot.curr_state == "MOVE_LEFTMOST_BLOCK":
            fr, ret = _robot.move_to_leftmost_block(frame)
        elif _robot.curr_state == "GRIP_PICKUP":
            _robot.grip_pickup()
        elif _robot.curr_state == "MOVE_OUR_CLOSET":
            fr, ret = _robot.move_to_xy(OUR_CLOSET_PICKUP[0], OUR_CLOSET_PICKUP[1], 90, "OUR_CLOSET", avoid_obstacles=True, frame=frame)
        elif _robot.curr_state == "MOVE_OUR_ROOM":
            fr, ret = _robot.move_to_xy(OUR_ROOM_MOVE[0], OUR_ROOM_MOVE[1], 0, "OUR_ROOM")
        elif _robot.curr_state == "MOVE_HALLWAY":
            fr, ret = _robot.move_to_xy(HALLWAY_MOVE[0], HALLWAY_MOVE[1], 0, "HALLWAY")
        elif _robot.curr_state == "MOVE_THEIR_ROOM":
            fr, ret = _robot.move_to_xy(THEIR_ROOM_MOVE[0], THEIR_ROOM_MOVE[1], -90, "THEIR_ROOM")
        elif _robot.curr_state == "MOVE_THEIR_CLOSET":
            fr, ret = _robot.move_to_xy(THEIR_CLOSET_PICKUP[0], THEIR_CLOSET_PICKUP[1], -90, "THEIR_CLOSET")
        elif _robot.curr_state == "AVOID_OBSTACLE_APRILTAG":
            ret = _robot.avoid_obstacle("APRILTAG", frame)
        elif _robot.curr_state == "AVOID_OBSTACLE_ROBOT":
            ret = _robot.avoid_obstacle("ROBOT", frame)
        elif _robot.curr_state == "AVOID_OBSTACLE_IR_FALLBACk":
            ret = _robot.avoid_obstacle("IR_SENSOR", frame)
        elif _robot.curr_state == "UPDATE_STATE":
            fr, ret = _robot.update_state_with_detections(_robot.minimax_agent.curr_state.our_position)
        elif _robot.curr_state == "DONE":
            state_done_flag = True
        
        
        # fr, ret = _robot.move_to_leftmost_block(frame)
        # if ret == 1:
        #     break
        
        ax.plot(position_history_x, position_history_y) 
        plt.draw()
        plt.pause(0.01)

        if fr is not None:
            cv2.imshow("img", fr)
            key = cv2.waitKey(1)
            if key == ord('z'):
                break

    _robot.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
    ep_robot.close()
    exit(0)