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

# vision = Vision('')

FEET_TO_METER_DIV_BY = 3.281

x = -1
y = -1
OUR_CLOSET_BOUNDARY = [(x,y), (x,y)]
OUR_ROOM_BOUNDARY = [(x,y), (x,y)]
HALLWAY_BOUNDARY = [(x,y), (x,y)]
THEIR_ROOM_BOUNDARY = [(x,y), (x,y)]
THEIR_CLOSET_BOUNDARY = [(x,y), (x,y)]

class State():
        
    def __init__(self):

        # initial conditions
        self.our_position = "OUR_ROOM"
        self.their_position = "THEIR_ROOM"
        self.our_held_block = None
        self.their_held_block = None
        self.block_positions = [("2x2", "OUR_CLOSET"), ("2x4", "OUR_CLOSET"), ("4x4", "OUR_CLOSET"), ("2x2", "THEIR_CLOSET"), ("2x4", "THEIR_CLOSET"), ("4x4", "THEIR_CLOSET")]
        self.current_score = None
        self.update_and_get_score()

    def copy(self):
        new_state = State()
        new_state.our_position = self.our_position
        new_state.their_position = self.their_position
        new_state.our_held_block = self.our_held_block
        new_state.their_held_block = self.their_held_block
        new_state.block_positions = self.block_positions.copy()
        return new_state

    def __repr__(self):
        return f"STATE:\
                \n\tOUR POS: {self.our_position}\
                \n\tTHEIR POS: {self.their_position}\
                \n\tOUR HELD BLOCK: {self.our_held_block}\
                \n\tTHEIR HELD BLOCK: {self.their_held_block}\
                \n\tBLOCK POSITIONS: {self.block_positions}\
                \n\tCURRENT SCORE: {self.current_score}"

    def update_and_get_score(self):
        
        score_change = 0
        
        for bp in self.block_positions:
            if bp[1] == "OUR_CLOSET":
                if bp[0] == "2x2":
                    score_change -= 16
                elif bp[0] == "2x4":
                    score_change -= 8
                elif bp[0] == "4x4":
                    score_change -= 4
            elif bp[1] == "OUR_ROOM":
                if bp[0] == "2x2":
                    score_change -= 4
                elif bp[0] == "2x4":
                    score_change -= 2
                elif bp[0] == "4x4":
                    score_change -= 1
            elif bp[1] == "THEIR_ROOM":
                if bp[0] == "2x2":
                    score_change += 4
                elif bp[0] == "2x4":
                    score_change += 2
                elif bp[0] == "4x4":
                    score_change += 1
            elif bp[1] == "THEIR_CLOSET":
                if bp[0] == "2x2":
                    score_change += 16
                elif bp[0] == "2x4":
                    score_change += 8
                elif bp[0] == "4x4":
                    score_change += 4
        
        if self.our_position == "OUR_CLOSET":
            if self.our_held_block == "2x2":
                score_change -= 16*0.75
            elif self.our_held_block == "2x4":
                score_change -= 8*0.75
            elif self.our_held_block == "4x4":
                score_change -= 4*0.75
        elif self.our_position == "OUR_ROOM":
            if self.our_held_block == "2x2":
                score_change -= 4*0.75
            elif self.our_held_block == "2x4":
                score_change -= 2*0.75
            elif self.our_held_block == "4x4":
                score_change -= 1*0.75
        elif self.our_position == "HALLWAY":
            if self.our_held_block == "2x2":
                score_change -= 16
            elif self.our_held_block == "2x4":
                score_change -= 8
            elif self.our_held_block == "4x4":
                score_change -= 4
        elif self.our_position == "THEIR_ROOM":
            if self.our_held_block == "2x2":
                score_change += 4*0.75
            elif self.our_held_block == "2x4":
                score_change += 2*0.75
            elif self.our_held_block == "4x4":
                score_change += 1*0.75
        elif self.our_position == "THEIR_CLOSET":
            if self.our_held_block == "2x2":
                score_change += 16*0.75
            elif self.our_held_block == "2x4":
                score_change += 8*0.75
            elif self.our_held_block == "4x4":
                score_change += 4*0.75

        if self.their_position == "OUR_CLOSET":
            if self.their_held_block == "2x2":
                score_change -= 16*0.75
            elif self.their_held_block == "2x4":
                score_change -= 8*0.75
            elif self.their_held_block == "4x4":
                score_change -= 4*0.75
        elif self.their_position == "OUR_ROOM":
            if self.their_held_block == "2x2":
                score_change -= 4*0.75
            elif self.their_held_block == "2x4":
                score_change -= 2*0.75
            elif self.their_held_block == "4x4":
                score_change -= 1*0.75
        elif self.their_position == "HALLWAY":
            if self.their_held_block == "2x2":
                score_change += 16*0.75
            elif self.their_held_block == "2x4":
                score_change += 8*0.75
            elif self.their_held_block == "4x4":
                score_change += 4*0.75
        elif self.their_position == "THEIR_ROOM":
            if self.their_held_block == "2x2":
                score_change += 4*0.75
            elif self.their_held_block == "2x4":
                score_change += 2*0.75
            elif self.their_held_block == "4x4":
                score_change += 1*0.75
        elif self.their_position == "THEIR_CLOSET":
            if self.their_held_block == "2x2":
                score_change += 16*0.75
            elif self.their_held_block == "2x4":
                score_change += 8*0.75
            elif self.their_held_block == "4x4":
                score_change += 4*0.75
        
        self.current_score = score_change
        return self.current_score
    
    def get_available_actions(self, us=True):
        
        action_list = []

        if us:
            curr_pos = self.our_position
            curr_held = self.our_held_block
        else:
            curr_pos = self.their_position
            curr_held = self.their_held_block

        if curr_pos == "OUR_CLOSET":
            action_list.append(Action.MOVE_OUR_ROOM)
            if curr_held is None:
                if any(bp[1] == "OUR_CLOSET" and bp[0] == "2x2" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x2)
                if any(bp[1] == "OUR_CLOSET" and bp[0] == "2x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x4)
                if any(bp[1] == "OUR_CLOSET" and bp[0] == "4x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_4x4)
            else:
                action_list.append(Action.DROP_BLOCK)
        elif curr_pos == "OUR_ROOM":
            action_list.append(Action.MOVE_OUR_CLOSET)
            action_list.append(Action.MOVE_HALLWAY)
            if curr_held is None:
                if any(bp[1] == "OUR_ROOM" and bp[0] == "2x2" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x2)
                if any(bp[1] == "OUR_ROOM" and bp[0] == "2x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x4)
                if any(bp[1] == "OUR_ROOM" and bp[0] == "4x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_4x4)
            else:
                action_list.append(Action.DROP_BLOCK)
        elif curr_pos == "HALLWAY":
            action_list.append(Action.MOVE_OUR_ROOM)
            action_list.append(Action.MOVE_THEIR_ROOM)
        elif curr_pos == "THEIR_ROOM":
            action_list.append(Action.MOVE_HALLWAY)
            action_list.append(Action.MOVE_THEIR_CLOSET)
            if curr_held is None:
                if any(bp[1] == "THEIR_ROOM" and bp[0] == "2x2" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x2)
                if any(bp[1] == "THEIR_ROOM" and bp[0] == "2x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x4)
                if any(bp[1] == "THEIR_ROOM" and bp[0] == "4x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_4x4)
            else:
                action_list.append(Action.DROP_BLOCK)
        elif curr_pos == "THEIR_CLOSET":
            action_list.append(Action.MOVE_THEIR_ROOM)
            if curr_held is None:
                if any(bp[1] == "THEIR_CLOSET" and bp[0] == "2x2" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x2)
                if any(bp[1] == "THEIR_CLOSET" and bp[0] == "2x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x4)
                if any(bp[1] == "THEIR_CLOSET" and bp[0] == "4x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_4x4)
            else:
                action_list.append(Action.DROP_BLOCK)
        
        return action_list
    
    def get_next_state(self, action, us=True):

        next_state = self.copy()

        if us:
            if action == Action.PICKUP_BLOCK_2x2:
                next_state.block_positions.remove(("2x2", self.our_position))
                next_state.our_held_block = "2x2"
            elif action == Action.PICKUP_BLOCK_2x4:
                next_state.block_positions.remove(("2x4", self.our_position))
                next_state.our_held_block = "2x4"
            elif action == Action.PICKUP_BLOCK_4x4:
                next_state.block_positions.remove(("4x4", self.our_position))
                next_state.our_held_block = "4x4"
            elif action == Action.DROP_BLOCK:
                next_state.our_held_block = None
                next_state.block_positions.append((self.our_held_block, self.our_position))
            elif action == Action.MOVE_OUR_CLOSET:
                next_state.our_position = "OUR_CLOSET"
            elif action == Action.MOVE_OUR_ROOM:
                next_state.our_position = "OUR_ROOM"
            elif action == Action.MOVE_HALLWAY:
                next_state.our_position = "HALLWAY"
            elif action == Action.MOVE_THEIR_ROOM:
                next_state.our_position = "THEIR_ROOM"
            elif action == Action.MOVE_THEIR_CLOSET:
                next_state.our_position = "THEIR_CLOSET"

        else:
            if action == Action.PICKUP_BLOCK_2x2:
                next_state.block_positions.remove(("2x2", self.their_position))
                next_state.their_held_block = "2x2"
            elif action == Action.PICKUP_BLOCK_2x4:
                next_state.block_positions.remove(("2x4", self.their_position))
                next_state.their_held_block = "2x4"
            elif action == Action.PICKUP_BLOCK_4x4:
                next_state.block_positions.remove(("4x4", self.their_position))
                next_state.their_held_block = "4x4"
            elif action == Action.DROP_BLOCK:
                next_state.their_held_block = None
                next_state.block_positions.append((self.our_held_block, self.their_position))
            elif action == Action.MOVE_OUR_CLOSET:
                next_state.their_position = "OUR_CLOSET"
            elif action == Action.MOVE_OUR_ROOM:
                next_state.their_position = "OUR_ROOM"
            elif action == Action.MOVE_HALLWAY:
                next_state.their_position = "HALLWAY"
            elif action == Action.MOVE_THEIR_ROOM:
                next_state.their_position = "THEIR_ROOM"
            elif action == Action.MOVE_THEIR_CLOSET:
                next_state.their_position = "THEIR_CLOSET"

        next_state.update_and_get_score()
        return next_state
        
# choose the action with the highest score
class ReflexAgent():
    def __init__(self, curr_state: State):
        self.curr_state: State = curr_state

    def choose_action(self):
        actions = self.curr_state.get_available_actions()

        best_action = None
        best_score = float('-inf')
        for action in actions:
            next_state = self.curr_state.get_next_state(action, us=True)
            if next_state.current_score > best_score:
                best_action = action
                best_score = next_state.current_score
        
        return best_action
    
    def perform_action(self, action):
        self.curr_state = self.curr_state.get_next_state(action, us=True)

class MiniMaxAgent():
    def __init__(self, curr_state: State, search_depth: int):
        self.curr_state: State = curr_state
        self.search_depth: int = search_depth

    def min_layer(self, curr_state: State, depth: int):
        if depth + 1 >= self.search_depth:
            return curr_state.current_score, None
        min_score = float('inf')
        min_action = None
        actions = curr_state.get_available_actions(us=False)
        for action in actions:
            next_state = curr_state.get_next_state(action, us=False)
            next_score, _ = self.max_layer(next_state, depth + 1)
            if next_score < min_score:
                min_score = next_score
                min_action = action
        return min_score, min_action
    
    def max_layer(self, curr_state: State, depth: int):
        if depth + 1 >= self.search_depth:
            return curr_state.current_score, None
        max_score = float('-inf')
        max_action = None
        actions = curr_state.get_available_actions(us=True)
        for action in actions:
            next_state = curr_state.get_next_state(action, us=True)
            next_score, _ = self.min_layer(next_state, depth + 1)
            if next_score > max_score:
                max_score = next_score
                max_action = action
        return max_score, max_action
        
    def choose_action(self):
        best_score, best_action = self.max_layer(self.curr_state, 0)
        print(f"found best action {best_action} with best score {best_score}")
        return best_action
    
    def perform_action(self, action):
        self.curr_state = self.curr_state.get_next_state(action)

class ExpectiMaxAgent():
    def __init__(self, curr_state: State, search_depth: int):
        self.curr_state: State = curr_state
        self.search_depth: int = search_depth

    def expecti_layer(self, curr_state: State, depth: int):
        if depth + 1 >= self.search_depth:
            return curr_state.current_score, None
        avg_score = 0.0
        avg_action = None
        actions = curr_state.get_available_actions(us=False)
        if len(actions) < 1:
            return 0.0, None
        for action in actions:
            next_state = curr_state.get_next_state(action, us=False)
            next_score, _ = self.max_layer(next_state, depth + 1)
            avg_score += next_score
        return avg_score/len(actions), None
    
    def max_layer(self, curr_state: State, depth: int):
        if depth + 1 >= self.search_depth:
            return curr_state.current_score, None
        max_score = float('-inf')
        max_action = None
        actions = curr_state.get_available_actions(us=True)
        for action in actions:
            next_state = curr_state.get_next_state(action, us=True)
            next_score, _ = self.expecti_layer(next_state, depth + 1)
            if next_score > max_score:
                max_score = next_score
                max_action = action
        return max_score, max_action
        
    def choose_action(self):
        best_score, best_action = self.max_layer(self.curr_state, 0)
        print(f"found best action {best_action} with best score {best_score}")
        return best_action
    
    def perform_action(self, action):
        self.curr_state = self.curr_state.get_next_state(action)

class Action(Enum):
    PICKUP_BLOCK_2x2 = 0,
    PICKUP_BLOCK_2x4 = 1,
    PICKUP_BLOCK_4x4 = 2,
    DROP_BLOCK = 3,
    MOVE_OUR_CLOSET = 10,
    MOVE_OUR_ROOM = 11,
    MOVE_HALLWAY = 12,
    MOVE_THEIR_ROOM = 13,
    MOVE_THEIR_CLOSET = 14,

# ep_robot = robot.Robot()
# ep_robot.initialize(conn_type="sta", sn="3JKCH7T001008H")
# ep_chassis = ep_robot.chassis
# ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
# ep_camera = ep_robot.camera
# ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
# ep_arm = ep_robot.robotic_arm
# ep_gripper = ep_robot.gripper
# ep_sensor = ep_robot.sensor

# # Start printing the gripper position
# ep_arm.sub_position(freq=5, callback=sub_data_handler)
# ep_chassis.sub_position(cs=1, freq=5, callback=chassis_subpos_cb)

# controller = IBVS_Controller(control_mode='2xz', interaction_mode='mean', num_pts=4)
# # controller.set_lambda_matrix([5.0, 50.0, 1.75, 1000.0]) # robot y velocity; robot z velocity; robot x velocity; robot z angular velocity
# controller.set_lambda_matrix([3.0, 1.25]) # robot y velocity; robot x velocity
# controller.set_desired_points([(-0.2, -0.7, 0.18), (0.2, -0.7, 0.18), (-0.2, 1.0, 0.18), (0.2, 1.0, 0.18)])

if __name__ == "__main__":
    init_state = State()
    # print(init_state)

    minimax_agent = MiniMaxAgent(init_state, 2*2)

    for i in range(12):
        best_action = minimax_agent.choose_action()
        minimax_agent.perform_action(best_action)
        print(minimax_agent.curr_state)

    # reflex_agent = ReflexAgent(init_state)

    # best_action = reflex_agent.choose_action()
    # print(best_action)
    # reflex_agent.perform_action(best_action)

    # best_action = reflex_agent.choose_action()
    # print(best_action)
    # reflex_agent.perform_action(best_action)

    # next_state = State()
    # print(next_state)
    # actions = next_state.get_available_actions(us=True)
    # print(f"curr actions: {actions}")
    
    # next_state = next_state.get_next_state(action=actions[0], us=True)
    # print(next_state)
    # actions = next_state.get_available_actions(us=True)
    # print(f"curr actions: {actions}")

    # next_state = next_state.get_next_state(action=actions[1], us=True)
    # print(next_state)
    # actions = next_state.get_available_actions(us=True)
    # print(f"curr actions: {actions}")

    # next_state = next_state.get_next_state(action=actions[0], us=True)
    # print(next_state)
    # actions = next_state.get_available_actions(us=True)
    # print(f"curr actions: {actions}")

    # next_state = next_state.get_next_state(action=actions[1], us=True)
    # print(next_state)
    # actions = next_state.get_available_actions(us=True)
    # print(f"curr actions: {actions}")

    # next_state = next_state.get_next_state(action=actions[1], us=True)
    # print(next_state)
    # actions = next_state.get_available_actions(us=True)
    # print(f"curr actions: {actions}")

    # next_state = next_state.get_next_state(action=actions[1], us=True)
    # print(next_state)
    # actions = next_state.get_available_actions(us=True)
    # print(f"curr actions: {actions}")

    # next_state = next_state.get_next_state(action=actions[1], us=True)
    # print(next_state)
    # actions = next_state.get_available_actions(us=True)
    # print(f"curr actions: {actions}")

    # next_state = next_state.get_next_state(action=actions[1], us=True)
    # print(next_state)
    # actions = next_state.get_available_actions(us=True)
    # print(f"curr actions: {actions}")