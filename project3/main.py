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
from state import *
from agents import *

# vision = Vision(r"C:\Users\jesto\Desktop\CMSC477\CMSC477-S25\runs\detect\train2\weights\best.pt")

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

    for i in range(30):
        best_action = minimax_agent.choose_action()
        minimax_agent.perform_action(best_action)
        # print(minimax_agent.curr_state)

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