from util import *
from state import *
import cv2
from robomaster import robot
from robomaster import camera
from queue import Empty
import time

class Robot():
    def __init__(self, ep_robot):
        self.ep_robot = ep_robot
        self.ep_chassis = ep_robot.chassis
        self.ep_chassis.sub_position(cs=0, freq=5, callback=self.chassis_callback)

    def chassis_callback(self, pos):
        x, y, z = pos
        # print(f"x: {x} y: {y} z: {z}")
        self.our_position = (float(x)+1.0, -float(y-1.0))
        print(f"current position: {self.our_position}")

    def attitude_callback(self, pos):
        yaw, pitch, roll = pos
        print(f"yaw: {yaw} pitch: {pitch} roll: {roll}")

    