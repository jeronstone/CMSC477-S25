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
IR_AVOID_THRESH = 375
IR_SAFE_THRESH = 400
PICKUP_TIMER_ABORT = 150
AVOID_POST_TIME_BUFFER = 10
APRILTAG_IN_THE_WAY_BUFFER = 0.35
APRILTAG_OBSTACLE_OFFSET = 0.5

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
        K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel
        marker_size_m = 0.153 # Size of the AprilTag in meters
        self.apriltag_detector = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)

        # minimax agent
        self.minimax_agent = MiniMaxAgent(State(), 2*2)
        self.curr_action = None
        self.curr_state = "DONE"#"MOVE_LEFTMOST_BLOCK"
        self.prev_state  = "DONE"
        self.state_timer = 0

        self.avoid_time_buffer = 0
        
        self.apriltag_map = {}
        self.apriltag_map["OUR_CLOSET"] = {}
        self.apriltag_map["OUR_ROOM"] = {}
        self.apriltag_map["THEIR_CLOSET"] = {}
        self.apriltag_map["THEIR_ROOM"] = {}
        
        # map controller
        #self.map = MapController(ep_robot)

        # IBVS controller
        self.controller = IBVS_Controller(control_mode='2xz', interaction_mode='mean', num_pts=4)
        self.controller.set_lambda_matrix([1.75, 0.5]) # robot y velocity; robot x velocity
        self.controller.set_desired_points([(-0.16, 0.375, 0.18), (0.16, 0.375, 0.18), (-0.16, 0.95, 0.18), (0.16, 0.95, 0.18)])
        
        self.ep_arm.moveto(x=200, y=-25).wait_for_completed()
        time.sleep(1.0)

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
            #print(self.get_current_location())
        except Exception as e:
            print(f"chassis callback exception: {e}")

    def dist_callback(self, dist):
        self.curr_ir_dist = dist[0]

    def get_current_location(self):
        if OUR_CLOSET_BOUNDARY[0][0] <= self.world_position[0] <= OUR_CLOSET_BOUNDARY[1][0] and OUR_CLOSET_BOUNDARY[0][1] <= self.world_position[1] <= OUR_CLOSET_BOUNDARY[1][1]:
            return "OUR_CLOSET"
        elif OUR_ROOM_BOUNDARY[0][0] <= self.world_position[0] <= OUR_ROOM_BOUNDARY[1][0] and OUR_ROOM_BOUNDARY[0][1] <= self.world_position[1] <= OUR_ROOM_BOUNDARY[1][1]:
            return "OUR_ROOM"
        elif HALLWAY_BOUNDARY[0][0] <= self.world_position[0] <= HALLWAY_BOUNDARY[1][0] and HALLWAY_BOUNDARY[0][1] <= self.world_position[1] <= HALLWAY_BOUNDARY[1][1]:
            return "HALLWAY"
        elif THEIR_ROOM_BOUNDARY[0][0] <= self.world_position[0] <= THEIR_ROOM_BOUNDARY[1][0] and THEIR_ROOM_BOUNDARY[0][1] <= self.world_position[1] <= THEIR_ROOM_BOUNDARY[1][1]:
            return "THEIR_ROOM"
        elif THEIR_CLOSET_BOUNDARY[0][0] <= self.world_position[0] <= THEIR_CLOSET_BOUNDARY[1][0] and THEIR_CLOSET_BOUNDARY[0][1] <= self.world_position[1] <= THEIR_CLOSET_BOUNDARY[1][1]:
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
            
        self.state_timer += 1
        
        if self.state_timer > PICKUP_TIMER_ABORT:
            print("taking too long, abort pickup...")
            self.prev_state = self.curr_state
            self.curr_state = "MOVE_OUR_CLOSET"
            return None, 1
            
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
                # print(f'No leftmost block detected')
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
                # print('close to block, transition')
                self.prev_state = self.curr_state
                self.curr_state = "GRIP_PICKUP"
                return fr, 1
            # print(f"horiz_ang: {most_horizontal_angle} depth: {depth} err_nrm: {err_nrm} vels: x {robot_x_velocity} y {robot_y_velocity}")
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
                # print('close to block, transition')
            # print(f"horiz_ang: {most_horizontal_angle} depth: {depth} err_nrm: {err_nrm} vels: x {robot_x_velocity} y {robot_y_velocity}")
                pass
    
    def get_avoid_apriltag_waypoint(self, desired_x, desired_y):
        avoid_apriltag_waypoint = None
        for tag, pos in self.apriltag_map[self.get_current_location()].items():
            # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            dist_to_lineseg = math.abs((desired_y-self.world_position[1])*pos[0] - (desired_x-self.world_position[0])*pos[1] + desired_x*self.world_position[1] - desired_y*self.world_position[0])
            dist_to_lineseg /= math.sqrt((desired_y-self.world_position[1])**2 + (desired_x-self.world_position[0])**2)
            
            if dist_to_lineseg < APRILTAG_IN_THE_WAY_BUFFER:
                dist_vect_x, dist_vect_y = desired_x - self.world_position[0], desired_y[1] - self.world_position[1]
                lineseg_norm = math.hypot(dist_vect_x, dist_vect_y)
                unit_vec_x, unit_vec_y = -dist_vect_y / lineseg_norm, dist_vect_x / lineseg_norm                
                avoid_apriltag_waypoint = (pos[0] + unit_vec_x*APRILTAG_OBSTACLE_OFFSET, pos[1] + unit_vec_y*APRILTAG_OBSTACLE_OFFSET)
                break # assume only 1 in the way ?
        
        return avoid_apriltag_waypoint
    
    '''
    Moves to global position x, y on the map using simple p loop
    '''
    def move_to_xy(self, desired_x, desired_y, desired_heading, final_location, avoid_obstacles=False, apriltag_mem=False, frame=None):
        
        if (avoid_obstacles or apriltag_mem) and frame is None:
            return None, -1
        
        if avoid_obstacles:
            ret = self.get_avoid_apriltag_waypoint(desired_x, desired_y)
            if ret is not None:
                print(f'OBSTACLE IN PLANNED PATH - ADDING WAYPOINT {ret[0], ret[1]}')
                return self.move_to_xy(ret[0], ret[1], desired_heading, final_location, avoid_obstacles, apriltag_mem, frame)
        
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
            
            if avoid_obstacles:
                #print("detecing obstacles to avoid...")
                
                if self.curr_ir_dist < IR_AVOID_THRESH:
                    self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
                    self.prev_state = self.curr_state
                    self.curr_state = "AVOID_OBSTACLE_IR_FALLBACk"
                    return None, 0
                    
                    
                fr, detections = self.vision.get_yolo_pred(frame, hough=False, depth_len=ROBOT_SIZE_EST)
                for i, d in enumerate(detections):
                    cls, corners, depth, detected_block_lines_hough = d
                    if cls == 0: # robot detected
                        # print(f'ROBOT DEPTH: {depth}')
                        intheway = (depth < ROBOT_CLOSE_THRESH)
                        if intheway:
                            self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
                            self.prev_state = self.curr_state
                            self.curr_state = "AVOID_OBSTACLE_ROBOT"
                            return fr, 0
            
            if avoid_obstacles or apriltag_mem:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray.astype(np.uint8)

                detections = self.apriltag_detector.find_tags(gray)
                intheway=False
                # print(f"apriltag detector detected: {len(detections)} apriltags")
                for detection in detections:
                    t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
                    distance = np.linalg.norm(t_ca-np.array([0, 0, APRILTAG_SIZE]))
                    
                    T_ca = np.array([[R_ca[0,0], R_ca[0,1], R_ca[0,2], t_ca[0]], 
                                     [R_ca[1,0], R_ca[1,1], R_ca[1,2], t_ca[1]],
                                     [R_ca[2,0], R_ca[2,1], R_ca[2,2], t_ca[2]],
                                     [        0,         0,         0,       1]])
                    
                    T_wa = self.T_w_bt @ T_ca
                    
                    t_wa_x = T_wa[0, 3]
                    t_wa_y = T_wa[1, 3]
                    
                    self.apriltag_map[self.get_current_location()][detection.tag_id] = (t_wa_x, t_wa_y)
                    
                    # print(f'Apriltag dist: {distance}')
                    if distance < APRILTAG_CLOSE_TRESH:
                        self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
                        self.prev_state = self.curr_state
                        self.curr_state = "AVOID_OBSTACLE_APRILTAG"
                        intheway=True
                
                if intheway:
                    return fr, 0
            
            return None, 1
        else:
            self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
            time.sleep(0.1)
            self.ep_chassis.move(x=0, y=0, z=-int(np.rad2deg(self.world_heading) - desired_heading), z_speed=45).wait_for_completed(2.0)
            time.sleep(2.0)
            #self.set_frame_rotation(desired_heading)
            self.minimax_agent.curr_state.world_position = final_location
            self.prev_state = self.curr_state
            self.curr_state = "UPDATE_STATE"
            return None, 0
        
    def avoid_obstacle(self, object, frame):
        # print(f'Avoiding {object}')
        if object == "APRILTAG":
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray.astype(np.uint8)

            detections = self.apriltag_detector.find_tags(gray)
            
            if len(detections) != 0:
                for detection in detections:
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
                            self.ep_chassis.drive_speed(x=0.0, y=-0.5, z=0.0, timeout=5)
                        else:               # left side, move right
                            self.ep_chassis.drive_speed(x=0.0, y=0.5, z=0.0, timeout=5)
                        
                        return 1
            
            # print("Obstacle avoided")
            # at this point, all detections were greater than thresh, or there were 0 detections
            self.curr_state = "AVOID_POST_TIME_BUFFER"
            self.avoid_time_buffer = 0
            return 0
                
        elif object == "ROBOT":
            self.ep_chassis.drive_speed(x=0.0, y=-0.5, z=0.0, timeout=5)
            fr, detections = self.vision.get_yolo_pred(frame, hough=False, depth_len=ROBOT_SIZE_EST)
            for i, d in enumerate(detections):
                cls, corners, depth, detected_block_lines_hough = d
                if cls == 0: # robot detected
                    intheway = (depth < ROBOT_CLOSE_THRESH)
                    if intheway:
                        # print("Theres a robot thats too close still")
                        
                        if corners[0] > 0:  # right side, move left
                            self.ep_chassis.drive_speed(x=0.0, y=-0.5, z=0.0, timeout=5)
                        else:               # left side, move right
                            self.ep_chassis.drive_speed(x=0.0, y=0.5, z=0.0, timeout=5)
                        
                        return 1
            
            # print("Obstacle avoided")
            self.curr_state = "AVOID_POST_TIME_BUFFER"
            self.avoid_time_buffer = 0
            return 0
        elif object == "IR_SENSOR":
            if self.curr_ir_dist < IR_SAFE_THRESH:
                # drive left slowly
                self.ep_chassis.drive_speed(x=0.0, y=-0.3, z=0.0, timeout=5)
                return 1
            else:
                # print("Obstacle avoided")
                self.curr_state = "AVOID_POST_TIME_BUFFER"
                self.avoid_time_buffer = 0
                return 0
        elif object == "TIME_BUFFER":
            if self.avoid_time_buffer < AVOID_POST_TIME_BUFFER:
                print(f'AVOID BUFFER {self.avoid_time_buffer}')
                self.avoid_time_buffer += 1
                return 1
            else:
                self.ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=5)
                self.curr_state = self.prev_state
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
            print(f"new action: {_robot.curr_action}")
            if _robot.curr_action == Action.PICKUP_BLOCK_2x2 or _robot.curr_action == Action.PICKUP_BLOCK_2x4 or _robot.curr_action == Action.PICKUP_BLOCK_4x4:
                _robot.curr_state = "MOVE_LEFTMOST_BLOCK"
                _robot.ep_arm.moveto(x=200, y=-50).wait_for_completed()
                _robot.state_timer = 0
                time.sleep(1.0)
            elif _robot.curr_action == Action.DROP_BLOCK:
                _robot.ep_arm.moveto(x=200, y=-50).wait_for_completed()
                _robot.curr_state = "GRIP_DROP"
            elif _robot.curr_action == Action.MOVE_OUR_CLOSET:
                _robot.ep_arm.moveto(x=200, y=-25).wait_for_completed()
                _robot.curr_state = "MOVE_OUR_CLOSET"
            elif _robot.curr_action == Action.MOVE_OUR_ROOM:
                _robot.ep_arm.moveto(x=200, y=-25).wait_for_completed()
                _robot.curr_state = "MOVE_OUR_ROOM"
            elif _robot.curr_action == Action.MOVE_HALLWAY:
                _robot.ep_arm.moveto(x=200, y=-25).wait_for_completed()
                _robot.curr_state = "MOVE_HALLWAY"
            elif _robot.curr_action == Action.MOVE_THEIR_ROOM:
                _robot.ep_arm.moveto(x=200, y=-25).wait_for_completed()
                _robot.curr_state = "MOVE_THEIR_ROOM"
            elif _robot.curr_action == Action.MOVE_THEIR_CLOSET:
                _robot.ep_arm.moveto(x=200, y=-25).wait_for_completed()
                _robot.curr_state = "MOVE_THEIR_CLOSET"
            else:
                _robot.curr_state = "DONE"
        
        print(f"curr state: {_robot.curr_state}")
        fr = None
        if _robot.curr_state == "MOVE_LEFTMOST_BLOCK":
            fr, ret = _robot.move_to_leftmost_block(frame)
            print(f'state timer: {_robot.state_timer}')
        elif _robot.curr_state == "GRIP_PICKUP":
            _robot.grip_pickup()
        elif _robot.curr_state == "GRIP_DROP":
            _robot.grip_drop()
        elif _robot.curr_state == "MOVE_OUR_CLOSET":
            fr, ret = _robot.move_to_xy(OUR_CLOSET_PICKUP[0], OUR_CLOSET_PICKUP[1], 0, "OUR_CLOSET", avoid_obstacles=True, apriltag_mem=True, frame=frame)
        elif _robot.curr_state == "MOVE_OUR_ROOM":
            fr, ret = _robot.move_to_xy(OUR_ROOM_MOVE[0], OUR_ROOM_MOVE[1], 90, "OUR_ROOM")
        elif _robot.curr_state == "MOVE_HALLWAY":
            fr, ret = _robot.move_to_xy(HALLWAY_MOVE[0], HALLWAY_MOVE[1], 90, "HALLWAY")
        elif _robot.curr_state == "MOVE_THEIR_ROOM":
            fr, ret = _robot.move_to_xy(THEIR_ROOM_MOVE[0], THEIR_ROOM_MOVE[1], -179, "THEIR_ROOM")
        elif _robot.curr_state == "MOVE_THEIR_CLOSET":
            fr, ret = _robot.move_to_xy(THEIR_CLOSET_PICKUP[0], THEIR_CLOSET_PICKUP[1], 180, "THEIR_CLOSET")
        elif _robot.curr_state == "AVOID_OBSTACLE_APRILTAG":
            ret = _robot.avoid_obstacle("APRILTAG", frame)
        elif _robot.curr_state == "AVOID_OBSTACLE_ROBOT":
            ret = _robot.avoid_obstacle("ROBOT", frame)
        elif _robot.curr_state == "AVOID_OBSTACLE_IR_FALLBACk":
            ret = _robot.avoid_obstacle("IR_SENSOR", frame)
        elif _robot.curr_state == "AVOID_POST_TIME_BUFFER":
            ret = _robot.avoid_obstacle("TIME_BUFFER", frame)
        elif _robot.curr_state == "UPDATE_STATE":
            fr, ret = _robot.update_state_with_detections(_robot.minimax_agent.curr_state.world_position)
        elif _robot.curr_state == "DONE":
            state_done_flag = True
        
        # fr, ret = _robot.move_to_leftmost_block(frame)
        # if ret == 1:
        #     break
        
        
        ax.plot(position_history_x, position_history_y, 'r') 
        plt.draw()
        plt.pause(0.01)
        
        
        if fr is not None:
            cv2.imshow("img", fr)
            key = cv2.waitKey(1)
            if key == ord('z'):
                break
        
        '''
        cv2.imshow("img", frame)
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
        _robot.ep_chassis.drive_speed(x=x_vel, y=y_vel, z=z_vel, timeout=5)
        '''

    _robot.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
    ep_robot.close()
    exit(0)