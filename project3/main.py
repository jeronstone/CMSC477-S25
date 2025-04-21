from enum import Enum
import math
from ultralytics import YOLO
import cv2
import time
import numpy as np
from robomaster import robot
from robomaster import camera
from ibvs_controller import *
from vision import *
from util import *

vision = Vision('')

class State():
        
    def __init__(self):
        self.curr_pos = ""
        self.their_pos = ""
        self.holding_block = False
        self.their_block = False
        self.block_positions = [] # (type, position)
        self.current_score = 0
    
    def update_and_get_score(self):
        
        score_change = 0
        
        for bp in self.block_positions:
            if bp[1] == "OUR_CLOSET":
                if bp[0] == "2x2":
                    score_change -= 16
                if bp[0] == "2x4":
                    score_change -= 8
                if bp[0] == "4x4":
                    score_change -= 4
            if bp[1] == "OUR_ROOM":
                if bp[0] == "2x2":
                    score_change -= 4
                if bp[0] == "2x4":
                    score_change -= 2
                if bp[0] == "4x4":
                    score_change -= 1
            if bp[1] == "THEIR_CLOSET":
                if bp[0] == "2x2":
                    score_change += 16
                if bp[0] == "2x4":
                    score_change += 8
                if bp[0] == "4x4":
                    score_change += 4
            if bp[1] == "THEIR_ROOM":
                if bp[0] == "2x2":
                    score_change += 4
                if bp[0] == "2x4":
                    score_change += 2
                if bp[0] == "4x4":
                    score_change += 1
        
        self.current_score = score_change
        return self.current_score
    
    def get_available_actions(self, us=True):
        
        action_list = []
        
        #if us:
        if self.curr_pos == "OUR_CLOSET":
            action_list.append(Action.MOVE_OUR_ROOM)
            if not self.holding_block:
                if any(bp[1] == "OUR_CLOSET" and bp[0] == "2x2" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x2)
                if any(bp[1] == "OUR_CLOSET" and bp[0] == "2x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x4)
                if any(bp[1] == "OUR_CLOSET" and bp[0] == "4x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_4x4)
            else:
                action_list.append(Action.DROP_BLOCK)
                    
        elif self.curr_pos == "OUR_ROOM":
            action_list.append(Action.MOVE_OUR_CLOSET)
            action_list.append(Action.MOVE_HALLWAY)
            if self.holding_block:
                action_list.append(Action.DROP_BLOCK)
            else:
                if any(bp[1] == "OUR_ROOM" and bp[0] == "2x2" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x2)
                if any(bp[1] == "OUR_ROOM" and bp[0] == "2x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x4)
                if any(bp[1] == "OUR_ROOM" and bp[0] == "4x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_4x4)
                    
        elif self.curr_pos == "HALLWAY":
            action_list.append(Action.MOVE_OUR_ROOM)
            action_list.append(Action.MOVE_THEIR_ROOM)
            
        elif self.curr_pos == "THEIR_ROOM":
            action_list.append(Action.MOVE_THEIR_CLOSET)
            action_list.append(Action.MOVE_HALLWAY)
            if self.holding_block:
                action_list.append(Action.DROP_BLOCK)
            else:
                if any(bp[1] == "THEIR_ROOM" and bp[0] == "2x2" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x2)
                if any(bp[1] == "THEIR_ROOM" and bp[0] == "2x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x4)
                if any(bp[1] == "THEIR_ROOM" and bp[0] == "4x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_4x4)
                    
        elif self.curr_pos == "THEIR_CLOSET":
            action_list.append(Action.MOVE_THEIR_ROOM)
            if self.holding_block:
                action_list.append(Action.DROP_BLOCK)
            else:
                if any(bp[1] == "THEIR_CLOSET" and bp[0] == "2x2" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x2)
                if any(bp[1] == "THEIR_CLOSET" and bp[0] == "2x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x4)
                if any(bp[1] == "THEIR_CLOSET" and bp[0] == "4x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_4x4)  
        
        return action_list

class Action(Enum):
    PICKUP_BLOCK_2x2 = 1,
    PICKUP_BLOCK_2x4 = 2,
    PICKUP_BLOCK_4x4 = 3,
    DROP_BLOCK = 4,
    MOVE_HALLWAY = 10,
    MOVE_OUR_ROOM = 11,
    MOVE_OUR_CLOSET = 12,
    MOVE_THEIR_ROOM = 13,
    MOVE_THEIR_CLOSET = 14,

ep_robot = robot.Robot()
ep_robot.initialize(conn_type="sta", sn="3JKCH7T001008H")
ep_chassis = ep_robot.chassis
ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
ep_arm = ep_robot.robotic_arm
ep_gripper = ep_robot.gripper
ep_sensor = ep_robot.sensor

# Start printing the gripper position
ep_arm.sub_position(freq=5, callback=sub_data_handler)

controller = IBVS_Controller(control_mode='2xz', interaction_mode='mean', num_pts=4)
# controller.set_lambda_matrix([5.0, 50.0, 1.75, 1000.0]) # robot y velocity; robot z velocity; robot x velocity; robot z angular velocity
controller.set_lambda_matrix([3.0, 1.25]) # robot y velocity; robot x velocity
controller.set_desired_points([(-0.2, -0.7, 0.18), (0.2, -0.7, 0.18), (-0.2, 1.0, 0.18), (0.2, 1.0, 0.18)])
