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

OP_MODE = "AUTO" # "MANUAL"

ROBOT_X_VELOCITY_MIN = -0.25
ROBOT_X_VELOCITY_MAX = 0.25
ROBOT_Y_VELOCITY_MIN = -0.25
ROBOT_Y_VELOCITY_MAX = 0.25
ROBOT_Z_VELOCITY_MIN = -1.0
ROBOT_Z_VELOCITY_MAX = 1.0
ROBOT_Z_POSITION_MIN = -50
ROBOT_Z_POSITION_MAX = -10
ROBOT_Z_ANGULAR_VELOCITY_MIN = -0.5
ROBOT_Z_ANGULAR_VELOCITY_MAX = 0.5

DIST_THRESH_X = 0.1
DIST_THRESH_Y = 0.1
APRILTAG_CLOSE_TRESH = 0.05
ROBOT_CLOSE_THRESH = 0.1
IR_AVOID_THRESH = 400
IR_SAFE_THRESH = 425
PICKUP_TIMER_ABORT = 50
AVOID_POST_TIME_BUFFER = 0

FEET_TO_METER_DIV_BY = 3.281

ROBOT_SIZE_EST = 0.25 #23 cm width, give it +2cm if its kinda sideways

# location boundaries
OUR_CLOSET_BOUNDARY = [(10.0/FEET_TO_METER_DIV_BY, 2.0/FEET_TO_METER_DIV_BY), (11.75/FEET_TO_METER_DIV_BY, 5.0/FEET_TO_METER_DIV_BY)]
OUR_ROOM_BOUNDARY = [(0.25/FEET_TO_METER_DIV_BY, 0.25/FEET_TO_METER_DIV_BY), (10.25/FEET_TO_METER_DIV_BY, 9.25/FEET_TO_METER_DIV_BY)]
HALLWAY_BOUNDARY = [(4.5/FEET_TO_METER_DIV_BY, 9.25/FEET_TO_METER_DIV_BY), (7.5/FEET_TO_METER_DIV_BY, 11.75/FEET_TO_METER_DIV_BY)]
THEIR_ROOM_BOUNDARY = [(1.75/FEET_TO_METER_DIV_BY, 11.75/FEET_TO_METER_DIV_BY), (11.75/FEET_TO_METER_DIV_BY, 20.75/FEET_TO_METER_DIV_BY)]
THEIR_CLOSET_BOUNDARY = [(0.25/FEET_TO_METER_DIV_BY, 16.0/FEET_TO_METER_DIV_BY), (2.0/FEET_TO_METER_DIV_BY, 19.0/FEET_TO_METER_DIV_BY)]

# pickup/dropoff locations
OUR_CLOSET_PICKUP = (2.62, 1.15)
OUR_CLOSET_DROPOFF = (3.0, 0.75) # increment y by 0.2 each time we drop off
OUR_ROOM_MOVE = (1.96, 1.08)
OUR_ROOM_PICKUP = (2.585, 1.110)
OUR_ROOM_DROPOFF = (1.5, 2.20) # decrement y by 0.2 each time we drop off
HALLWAY_MOVE = (1.96, 3.28)
THEIR_ROOM_MOVE = (1.96, 5.50)
THEIR_ROOM_PICKUP = (2.142, 5.426)
THEIR_ROOM_DROPOFF = (2.9, 4.1) # increment y by 0.2 each time we drop off
THEIR_CLOSET_PICKUP = (0.72, 5.37)
THEIR_CLOSET_DROPOFF = (1.512, 5.375) # increment y by 0.2 each time we drop off

LEGO_BIG_DESIRED = [(-0.16, 0.375, 0.18), (0.16, 0.375, 0.18), (-0.16, 0.95, 0.18), (0.16, 0.95, 0.18)]
LEGO_MEDIUM_DESIRED = [(-0.13, 0.55, 0.18), (0.13, 0.55, 0.18), (-0.13, 0.95, 0.18), (0.13, 0.95, 0.18)]
LEGO_SMALL_DESIRED = [(-0.1, 0.55, 0.18), (0.1, 0.55, 0.18), (-0.1, 0.95, 0.18), (0.1, 0.95, 0.18)]

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
        self.reported_position = (0.0, 0.0)
        self.world_position = (3.0/FEET_TO_METER_DIV_BY, 3.0/FEET_TO_METER_DIV_BY)

        self.prev_location = "OUR_CLOSET"
        self.curr_location = "OUR_ROOM"

        self.reported_heading = 0.0
        self.world_heading = 0.0
        
        self.T_w_b0 = np.array([[1,  0,  0, 3.0/FEET_TO_METER_DIV_BY],
                                [0, -1,  0, 3.0/FEET_TO_METER_DIV_BY],
                                [0,  0, -1,                        0],
                                [0,  0,  0,                        1]]) # initial body position in world frame
        self.T_b0_i = None # reported position in initial body frame
        self.T_i_bt = None # current body position in reported frame
        self.T_w_bt = None # current body position in world frame
        self.calculated_initial_matrices = False # whether we have calculated T_b0_i

        # vision controller
        self.vision = Vision(r"..\runs\detect\train2\weights\best.pt")
        _, _ = self.vision.get_yolo_pred(cv2.imread(r"project3/dummy.png")) # warm up[] model
        K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel
        marker_size_m = 0.153 # Size of the AprilTag in meters
        self.apriltag_detector = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)

        # minimax agent
        self.minimax_agent = MiniMaxAgent(State(), 2*2)
        self.curr_action = None
        self.curr_state = "DONE"#"MOVE_LEFTMOST_BLOCK"
        self.prev_state  = "DONE_WAIT"
        self.state_timer = 0

        self.avoid_time_buffer = 0
        
        # map controller
        #self.map = MapController(ep_robot)

        # IBVS controller
        self.controller = IBVS_Controller(control_mode='2xz', interaction_mode='mean', num_pts=4)
        self.controller.set_lambda_matrix([1.9, 0.5]) # robot y velocity; robot x velocity
        self.controller.set_desired_points(LEGO_BIG_DESIRED)
        
        self.ep_arm.moveto(x=200, y=-25).wait_for_completed(1.0)
        # time.sleep(1.0)

    def attitude_callback(self, pos):
        try:
            yaw, _, _ = pos
            self.reported_heading = yaw
            if self.calculated_initial_matrices == False:
                c, s = np.cos(np.deg2rad(self.reported_heading)), np.sin(np.deg2rad(self.reported_heading))
                T_i_b0 = np.array([[c, -s, 0, self.reported_position[0]],
                                   [s,  c, 0, self.reported_position[1]],
                                   [0,  0, 1,                         0],
                                   [0,  0, 0,                         1]])
                self.T_b0_i = np.linalg.inv(T_i_b0)
                # print(f"T_w_b0: {self.T_w_b0}")
                # print(f"T_b0_i: {self.T_b0_i}")
                self.calculated_initial_matrices = True
        except Exception as e:
            print(f"attitude callback exception: {e}")

    def chassis_callback(self, pos):
        try:
            x, y, _ = pos
            self.reported_position = (x, y)
            if self.calculated_initial_matrices == True:
                c, s = np.cos(np.deg2rad(self.reported_heading)), np.sin(np.deg2rad(self.reported_heading))
                self.T_i_bt = np.array([[c, -s, 0, self.reported_position[0]],
                                        [s,  c, 0, self.reported_position[1]],
                                        [0,  0, 1,                         0],
                                        [0,  0, 0,                         1]])
                # print(f"T_i_bt: {self.T_i_bt}")
                self.T_w_bt = self.T_w_b0 @ self.T_b0_i @ self.T_i_bt
                # print(f"T_w_bt: {T_w_bt}")
                self.world_position = (self.T_w_bt[0, 3], self.T_w_bt[1, 3])
                self.world_heading = np.arctan2(self.T_w_bt[1, 0], self.T_w_bt[0, 0])
                # print(f"world position: {self.world_position}; world heading: {self.world_heading}")
                position_history_x.append(self.world_position[0])
                position_history_y.append(self.world_position[1])
            #print(f"current position: {self.world_position}")
        except Exception as e:
            print(f"chassis callback exception: {e}")

    def dist_callback(self, dist):
        self.curr_ir_dist = dist[0]

    def get_location(self, x, y):
        if OUR_CLOSET_BOUNDARY[0][0] <= x <= OUR_CLOSET_BOUNDARY[1][0] and OUR_CLOSET_BOUNDARY[0][1] <= y <= OUR_CLOSET_BOUNDARY[1][1]:
            return "OUR_CLOSET"
        elif OUR_ROOM_BOUNDARY[0][0] <= x <= OUR_ROOM_BOUNDARY[1][0] and OUR_ROOM_BOUNDARY[0][1] <= y <= OUR_ROOM_BOUNDARY[1][1]:
            return "OUR_ROOM"
        elif HALLWAY_BOUNDARY[0][0] <= x <= HALLWAY_BOUNDARY[1][0] and HALLWAY_BOUNDARY[0][1] <= y <= HALLWAY_BOUNDARY[1][1]:
            return "HALLWAY"
        elif THEIR_ROOM_BOUNDARY[0][0] <= x <= THEIR_ROOM_BOUNDARY[1][0] and THEIR_ROOM_BOUNDARY[0][1] <= y <= THEIR_ROOM_BOUNDARY[1][1]:
            return "THEIR_ROOM"
        elif THEIR_CLOSET_BOUNDARY[0][0] <= x <= THEIR_CLOSET_BOUNDARY[1][0] and THEIR_CLOSET_BOUNDARY[0][1] <= y <= THEIR_CLOSET_BOUNDARY[1][1]:
            return "THEIR_CLOSET"
        else:
            return "undef"
        
    def update_state_with_detections(self, location: str, yolo_detections):
        
        if len(yolo_detections) > 0:
            block_list = [block for block in self.minimax_agent.curr_state.block_positions if block[1] != location] # remove blocks that were previously reported at this location
            for detection in yolo_detections:
                cls, _, _, _ = detection
                if cls == 0: # enemy robot
                    self.minimax_agent.curr_state.their_position = location
                elif 2 <= cls <= 4: # block
                    block_list.append(("4x4", location))
            self.minimax_agent.curr_state.block_positions = block_list

        self.curr_state = "DONE_WAIT"

    def grip_pickup(self):
            
        self.ep_chassis.move(x=0.05, y=0, z=0, xy_speed=0.2).wait_for_completed(2.0) # move slightly forward to position tower in gripper
        time.sleep(2.0)
        
        self.ep_gripper.close(power=125)
        time.sleep(1.0)
        self.ep_gripper.pause()
        
        self.ep_arm.moveto(x=200, y=-10).wait_for_completed(2.0)
        time.sleep(2.0)
        
        self.curr_state = "DONE_WAIT"
        
    def grip_drop(self):

        self.ep_arm.moveto(x=200, y=-50).wait_for_completed(2.0)
        time.sleep(2.0)
        
        self.ep_gripper.open(power=125)
        time.sleep(1.0)
        self.ep_gripper.pause()
        
        # time.sleep(2.0)

        self.curr_state = "DONE_WAIT"

    def move_to_leftmost_block_wait(self):
        self.curr_state = "MOVE_LEFTMOST_BLOCK"

    def update_wait(self):
        self.curr_state = "UPDATE_STATE"

    def done_wait(self):
        self.curr_state = "DONE"

    def move_to_leftmost_block(self, frame, yolo_detections):
        print('here')
        self.state_timer += 1
        
        # if self.state_timer >= PICKUP_TIMER_ABORT:
        #     print("taking too long, abort pickup...")
        #     self.prev_state = self.curr_state
        #     self.curr_state = "MOVE_OUR_CLOSET"
        #     self.curr_action = Action.MOVE_OUR_CLOSET
        #     return frame
        
        if len(yolo_detections) == 0:
            self.ep_chassis.drive_speed(x=0, y=0, z=20, timeout=5)
            return frame
        else:            
            leftmost = 999
            leftmost_idx = -1
            for i, d in enumerate(yolo_detections):
                cls, corners, depth, detected_block_lines_hough = d
                if cls >= 2 and cls <= 4: # if its a block
                    if corners[0] < leftmost: # if its more left than a previous block
                        leftmost = corners[0]
                        leftmost_idx = i
                        
            if leftmost_idx == -1:
                # print(f'No leftmost block detected')
                self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
                return frame
            
            cls, corners, depth, detected_block_lines_hough = yolo_detections[leftmost_idx]

            if cls == 2:
                self.controller.set_desired_points(LEGO_BIG_DESIRED)
            elif cls == 3:
                self.controller.set_desired_points(LEGO_MEDIUM_DESIRED)
            elif cls == 4:
                self.controller.set_desired_points(LEGO_SMALL_DESIRED)
                    
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
                # print('close to block, transition')
                self.prev_state = self.curr_state
                self.curr_state = "GRIP_PICKUP"
                return frame
            # print(f"horiz_ang: {most_horizontal_angle} depth: {depth} err_nrm: {err_nrm} vels: x {robot_x_velocity} y {robot_y_velocity}")
            return frame
    
    '''
    Moves to global position x, y on the map using simple p loop and constant speed
    '''
    def move_to_xy(self, frame, yolo_detections, apriltag_detections, desired_x, desired_y, desired_heading, final_location):
        
        err_x_w = self.world_position[0] - desired_x # x error in world frame
        err_y_w = self.world_position[1] - desired_y # y error in world frame

        if abs(err_x_w) > DIST_THRESH_X or abs(err_y_w) > DIST_THRESH_Y:
            
            velo_x_w = -0.5 * err_x_w # x vel in world frame
            velo_y_w = -0.5 * err_y_w # y vel in world frame
            
            # if abs(err_x_w) > DIST_THRESH_X:
            #     velo_x_w = -math.copysign(0.4, err_x_w)
                
            # if abs(err_y_w) > DIST_THRESH_Y:
            #     velo_y_w = math.copysign(0.4, err_y_w)
                
            # print(f'Error: {err_x_w} {err_y_w} \t Velos: {velo_x_w} {velo_y_w}')

            #velos_r = self.frame_rotation @ np.array([[float(velo_x_w)], [float(velo_y_w)]])

            #velo_x_r = clamp(velos_r.item(0), ROBOT_X_VELOCITY_MIN, ROBOT_X_VELOCITY_MAX)
            #velo_y_r = clamp(velos_r.item(1), ROBOT_Y_VELOCITY_MIN, ROBOT_Y_VELOCITY_MAX)

            R_bt_w = np.linalg.inv(self.T_w_bt)[:3, :3]
            velos_r = R_bt_w @ np.array([[velo_x_w], [velo_y_w], [0]])
            velo_x_r = clamp(velos_r.item(0), ROBOT_X_VELOCITY_MIN, ROBOT_X_VELOCITY_MAX)
            velo_y_r = clamp(velos_r.item(1), ROBOT_Y_VELOCITY_MIN, ROBOT_Y_VELOCITY_MAX)
            
            self.ep_chassis.drive_speed(x=velo_x_r, y=velo_y_r, z=0.0, timeout=5)
            #time.sleep(0.1)
            
            #print("detecing obstacles to avoid...")
            
            if self.curr_ir_dist < IR_AVOID_THRESH:
                self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
                self.prev_state = self.curr_state
                self.curr_state = "AVOID_OBSTACLE_IR_FALLBACk"
                return frame
                
            for i, d in enumerate(yolo_detections):
                cls, corners, depth, detected_block_lines_hough = d
                if cls == 0: # robot detected
                    # print(f'ROBOT DEPTH: {depth}')
                    intheway = (depth < ROBOT_CLOSE_THRESH)
                    if intheway:
                        self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
                        self.prev_state = self.curr_state
                        self.curr_state = "AVOID_OBSTACLE_ROBOT"
                        return frame

            # print(f"apriltag detector detected: {len(detections)} apriltags")
            for detection in apriltag_detections:
                t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
                distance = np.linalg.norm(t_ca-np.array([0, 0, APRILTAG_SIZE]))
                # print(f'Apriltag dist: {distance}')
                if distance < APRILTAG_CLOSE_TRESH:
                    self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
                    self.prev_state = self.curr_state
                    self.curr_state = "AVOID_OBSTACLE_APRILTAG"
                    return frame
            
            return frame
        else:
            self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
            # time.sleep(0.1)

            self.prev_location = self.curr_location
            self.curr_location = final_location

            if self.prev_location == "OUR_CLOSET":
                if self.curr_location == "OUR_ROOM":
                    desired_heading = 90
            elif self.prev_location == "OUR_ROOM":
                if self.curr_location == "OUR_CLOSET":
                    desired_heading = 0
                elif self.curr_location == "HALLWAY":
                    desired_heading = 90
            elif self.prev_location == "HALLWAY":
                if self.curr_location == "OUR_ROOM":
                    desired_heading = 90
                elif self.curr_location == "THEIR_ROOM":
                    desired_heading = -90
            elif self.prev_location == "THEIR_ROOM":
                if self.curr_location == "HALLWAY":
                    desired_heading = -90
                elif self.curr_location == "THEIR_CLOSET":
                    desired_heading = 180
            elif self.prev_location == "THEIR_CLOSET":
                if self.curr_location == "THEIR_ROOM":
                    desired_heading = -90

            turn_angle = ((desired_heading - np.rad2deg(_robot.world_heading) + 540) % 360) - 180
            # turn_angle = desired_heading - np.rad2deg(self.world_heading)
            self.ep_chassis.move(x=0, y=0, z=turn_angle, z_speed=60).wait_for_completed(2.0)
            # time.sleep(2.0)
            #self.set_frame_rotation(desired_heading)
            self.prev_state = self.curr_state
            if final_location == "HALLWAY":
                self.curr_state = "DONE_WAIT"
            else:
                self.curr_state = "UPDATE_WAIT"
            return frame
        
    def avoid_obstacle(self, frame, object, yolo_detections, apriltag_detections):
        # print(f'Avoiding {object}')
        if object == "APRILTAG":
            
            if len(apriltag_detections) != 0:
                for detection in apriltag_detections:
                    t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
                    distance = np.linalg.norm(t_ca-np.array([0, 0, APRILTAG_SIZE]))
                    # print(f'Apriltag dist: {distance}')
                    if distance < APRILTAG_CLOSE_TRESH:
                        # print("There's an Apriltag thats too close still")
                        
                        pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)
                        top_left = tuple(pts[0][0])  # First corner
                        # top_right = tuple(pts[1][0])  # Second corner
                        # bottom_right = tuple(pts[2][0])  # Third corner
                        # bottom_left = tuple(pts[3][0])  # Fourth corner
                        if top_left[0] > 0:    # right side, move left
                            self.ep_chassis.drive_speed(x=0.0, y=ROBOT_Y_VELOCITY_MIN, z=0.0, timeout=5)
                        else:               # left side, move right
                            self.ep_chassis.drive_speed(x=0.0, y=ROBOT_Y_VELOCITY_MAX, z=0.0, timeout=5)
                        
                        return frame
            
            # print("Obstacle avoided")
            # at this point, all detections were greater than thresh, or there were 0 detections
            self.curr_state = "AVOID_POST_TIME_BUFFER"
            self.avoid_time_buffer = 0
            return frame
                
        elif object == "ROBOT":
            self.ep_chassis.drive_speed(x=0.0, y=ROBOT_Y_VELOCITY_MIN, z=0.0, timeout=5)
            for i, d in enumerate(yolo_detections):
                cls, corners, depth, detected_block_lines_hough = d
                if cls == 0: # robot detected
                    intheway = (depth < ROBOT_CLOSE_THRESH)
                    if intheway:
                        # print("Theres a robot thats too close still")
                        
                        if corners[0] > 0:  # right side, move left
                            self.ep_chassis.drive_speed(x=0.0, y=ROBOT_Y_VELOCITY_MIN, z=0.0, timeout=5)
                        else:               # left side, move right
                            self.ep_chassis.drive_speed(x=0.0, y=ROBOT_Y_VELOCITY_MAX, z=0.0, timeout=5)
                        
                        return frame
            
            # print("Obstacle avoided")
            self.curr_state = "AVOID_POST_TIME_BUFFER"
            self.avoid_time_buffer = 0
            return frame
        elif object == "IR_SENSOR":
            if self.curr_ir_dist < IR_SAFE_THRESH:
                # drive left slowly
                self.ep_chassis.drive_speed(x=0.0, y=ROBOT_Y_VELOCITY_MIN, z=0.0, timeout=5)
                return frame
            else:
                # print("Obstacle avoided")
                self.curr_state = "AVOID_POST_TIME_BUFFER"
                self.avoid_time_buffer = 0
                return frame
        elif object == "TIME_BUFFER":
            if self.avoid_time_buffer < AVOID_POST_TIME_BUFFER:
                print(f'AVOID BUFFER {self.avoid_time_buffer}')
                self.avoid_time_buffer += 1
                return frame
            else:
                self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
                self.curr_state = self.prev_state
                return frame
        else:
            return frame



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

    frame = None
        
    while True:
        
        frame = None
        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.2)
        except Empty:
            time.sleep(0.001)
            continue

        if frame is None:
            time.sleep(0.001)
            continue

        frame, yolo_detections = _robot.vision.get_yolo_pred(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)
        apriltag_detections = _robot.apriltag_detector.find_tags(gray)

        if OP_MODE == "AUTO":
            if state_done_flag:
                print("done flag true")
                state_done_flag = False
                _robot.minimax_agent.perform_action(_robot.curr_action)
                _robot.curr_action = _robot.minimax_agent.choose_action()
                print(f"new action: {_robot.curr_action}")
                if _robot.curr_action == Action.PICKUP_BLOCK_2x2 or _robot.curr_action == Action.PICKUP_BLOCK_2x4 or _robot.curr_action == Action.PICKUP_BLOCK_4x4:
                    _robot.curr_state = "MOVE_LEFTMOST_BLOCK_WAIT"
                    _robot.ep_arm.moveto(x=200, y=-50).wait_for_completed(1.0)
                    # time.sleep(1.0)
                    _robot.state_timer = 0
                    print('curr action pickup')
                elif _robot.curr_action == Action.DROP_BLOCK:
                    _robot.ep_arm.moveto(x=200, y=-50).wait_for_completed(1.0)
                    # time.sleep(1.0)
                    _robot.curr_state = "GRIP_DROP"
                elif _robot.curr_action == Action.MOVE_OUR_CLOSET:
                    _robot.ep_arm.moveto(x=200, y=-25).wait_for_completed(1.0)
                    # time.sleep(1.0)
                    _robot.curr_state = "MOVE_OUR_CLOSET"
                elif _robot.curr_action == Action.MOVE_OUR_ROOM:
                    _robot.ep_arm.moveto(x=200, y=-25).wait_for_completed(1.0)
                    # time.sleep(1.0)
                    _robot.curr_state = "MOVE_OUR_ROOM"
                elif _robot.curr_action == Action.MOVE_HALLWAY:
                    _robot.ep_arm.moveto(x=200, y=-25).wait_for_completed(1.0)
                    # time.sleep(1.0)
                    _robot.curr_state = "MOVE_HALLWAY"
                elif _robot.curr_action == Action.MOVE_THEIR_ROOM:
                    _robot.ep_arm.moveto(x=200, y=-25).wait_for_completed(1.0)
                    # time.sleep(1.0)
                    _robot.curr_state = "MOVE_THEIR_ROOM"
                elif _robot.curr_action == Action.MOVE_THEIR_CLOSET:
                    _robot.ep_arm.moveto(x=200, y=-25).wait_for_completed(1.0)
                    # time.sleep(1.0)
                    _robot.curr_state = "MOVE_THEIR_CLOSET"
                else:
                    _robot.curr_state = "DONE_WAIT"
                
            
            print(f"curr state: {_robot.curr_state}")
            if _robot.curr_state == "MOVE_LEFTMOST_BLOCK":
                frame = _robot.move_to_leftmost_block(frame, yolo_detections)
                print(f'state timer: {_robot.state_timer}')
            elif _robot.curr_state == "GRIP_PICKUP":
                _robot.grip_pickup()
            elif _robot.curr_state == "GRIP_DROP":
                _robot.grip_drop()
            elif _robot.curr_state == "MOVE_OUR_CLOSET":
                frame = _robot.move_to_xy(frame, yolo_detections, apriltag_detections, OUR_CLOSET_PICKUP[0], OUR_CLOSET_PICKUP[1], 0, "OUR_CLOSET")
            elif _robot.curr_state == "MOVE_OUR_ROOM":
                frame = _robot.move_to_xy(frame, yolo_detections, apriltag_detections, OUR_ROOM_MOVE[0], OUR_ROOM_MOVE[1], 90, "OUR_ROOM")
            elif _robot.curr_state == "MOVE_HALLWAY":
                frame = _robot.move_to_xy(frame, yolo_detections, apriltag_detections, HALLWAY_MOVE[0], HALLWAY_MOVE[1], 90, "HALLWAY")
            elif _robot.curr_state == "MOVE_THEIR_ROOM":
                frame = _robot.move_to_xy(frame, yolo_detections, apriltag_detections, THEIR_ROOM_MOVE[0], THEIR_ROOM_MOVE[1], 180, "THEIR_ROOM")
            elif _robot.curr_state == "MOVE_THEIR_CLOSET":
                frame = _robot.move_to_xy(frame, yolo_detections, apriltag_detections, THEIR_CLOSET_PICKUP[0], THEIR_CLOSET_PICKUP[1], 180, "THEIR_CLOSET")
            elif _robot.curr_state == "AVOID_OBSTACLE_APRILTAG":
                frame = _robot.avoid_obstacle(frame, "APRILTAG", yolo_detections, apriltag_detections)
            elif _robot.curr_state == "AVOID_OBSTACLE_ROBOT":
                frame = _robot.avoid_obstacle(frame, "ROBOT", yolo_detections, apriltag_detections)
            elif _robot.curr_state == "AVOID_OBSTACLE_IR_FALLBACk":
                frame = _robot.avoid_obstacle(frame, "IR_SENSOR", yolo_detections, apriltag_detections)
            elif _robot.curr_state == "AVOID_POST_TIME_BUFFER":
                frame = _robot.avoid_obstacle(frame, "TIME_BUFFER", yolo_detections, apriltag_detections)
            elif _robot.curr_state == "UPDATE_STATE":
                _robot.update_state_with_detections(_robot.curr_location, yolo_detections)
                print(_robot.minimax_agent.curr_state)
            elif _robot.curr_state == "MOVE_LEFTMOST_BLOCK_WAIT":
                _robot.move_to_leftmost_block_wait()
            elif _robot.curr_state == "UPDATE_WAIT":
                _robot.update_wait()
            elif _robot.curr_state == "DONE_WAIT":
                _robot.done_wait()
            elif _robot.curr_state == "DONE":
                state_done_flag = True
        else:
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
            elif key == ord('r'):
                _robot.ep_arm.moveto(x=200, y=-25).wait_for_completed(1.0)
            elif key == ord('f'):
                _robot.ep_arm.moveto(x=200, y=-50).wait_for_completed(1.0)
            elif key == ord('l'):
                print('turn to 0 deg')
                turn_angle = ((0 - np.rad2deg(_robot.world_heading) + 540) % 360) - 180
                _robot.ep_chassis.move(x=0, y=0, z=turn_angle, z_speed=75).wait_for_completed(4.0)
            elif key == ord('i'):
                print('turn to 90 deg')
                turn_angle = ((90 - np.rad2deg(_robot.world_heading) + 540) % 360) - 180
                _robot.ep_chassis.move(x=0, y=0, z=turn_angle, z_speed=75).wait_for_completed(4.0)
            elif key == ord('j'):
                print('turn to 180 deg')
                turn_angle = ((180 - np.rad2deg(_robot.world_heading) + 540) % 360) - 180
                _robot.ep_chassis.move(x=0, y=0, z=turn_angle, z_speed=75).wait_for_completed(4.0)
            elif key == ord('k'):
                print('turn to -90 deg')
                turn_angle = ((-90 - np.rad2deg(_robot.world_heading) + 540) % 360) - 180
                _robot.ep_chassis.move(x=0, y=0, z=turn_angle, z_speed=75).wait_for_completed(4.0)
            elif key == ord('z'):
                break
            _robot.ep_chassis.drive_speed(x=x_vel, y=y_vel, z=z_vel, timeout=5)
            print(f"world_position: {_robot.world_position} world_heading: {_robot.world_heading}")
        
        if len(position_history_x) > 2048:
            position_history_x = position_history_x[-2048:]
        if len(position_history_y) > 2048:
            position_history_x = position_history_y[-2048:]
        if len(position_history_x) > len(position_history_y):
            position_history_x = position_history_x[-len(position_history_y):]
        if len(position_history_y) > len(position_history_x):
            position_history_y = position_history_y[-len(position_history_x):]
        ax.plot(position_history_x, position_history_y, 'r')
        plt.draw()
        plt.pause(0.01)
        
        if frame is not None:
            cv2.imshow("img", frame)
            key = cv2.waitKey(1)
            if key == ord('z'):
                break
        
        '''
        cv2.imshow("img", frame)
        
        '''

    plt.close()
    cv2.destroyAllWindows()
    _robot.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
    _robot.ep_arm.moveto(x=200, y=-50).wait_for_completed(1.0)
    _robot.ep_gripper.open(power=150)
    time.sleep(1.0)
    _robot.ep_gripper.pause()
    ep_robot.close()
    exit(0)

            
