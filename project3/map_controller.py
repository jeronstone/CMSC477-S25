from util import *
from state import *
from ibvs_controller import *

import cv2
from robomaster import robot
from robomaster import camera
from queue import Empty
import time
from ultralytics import YOLO
import numpy as np

model = YOLO(r"..\runs\detect\train2\weights\best.pt")

FEET_TO_METER_DIV_BY = 3.281

# initial estimates, need to measure and refine
# OUR_CLOSET_BOUNDARY = [(2.6, 0.60), (3.6, 1.5)]
# OUR_ROOM_BOUNDARY = [(0.25, 0.25), (2.6, 2.25)]
# HALLWAY_BOUNDARY = [(1.0, 2.5), (2.6, 4.5)]
# THEIR_ROOM_BOUNDARY = [(3.75/FEET_TO_METER_DIV_BY, 12.75/FEET_TO_METER_DIV_BY), (8.25/FEET_TO_METER_DIV_BY, 20.25/FEET_TO_METER_DIV_BY)]
# THEIR_CLOSET_BOUNDARY = [(0.75/FEET_TO_METER_DIV_BY, 15.75/FEET_TO_METER_DIV_BY), (2.25/FEET_TO_METER_DIV_BY, 20.5/FEET_TO_METER_DIV_BY)]

# location boundaries
OUR_CLOSET_BOUNDARY = [(10.0/FEET_TO_METER_DIV_BY, 2.0/FEET_TO_METER_DIV_BY), (11.75/FEET_TO_METER_DIV_BY, 5.0/FEET_TO_METER_DIV_BY)]
OUR_ROOM_BOUNDARY = [(0.25/FEET_TO_METER_DIV_BY, 0.25/FEET_TO_METER_DIV_BY), (10.25/FEET_TO_METER_DIV_BY, 9.25/FEET_TO_METER_DIV_BY)]
HALLWAY_BOUNDARY = [(4.5/FEET_TO_METER_DIV_BY, 9.25/FEET_TO_METER_DIV_BY), (7.5/FEET_TO_METER_DIV_BY, 11.75/FEET_TO_METER_DIV_BY)]
THEIR_ROOM_BOUNDARY = [(1.75/FEET_TO_METER_DIV_BY, 11.75/FEET_TO_METER_DIV_BY), (11.75/FEET_TO_METER_DIV_BY, 20.75/FEET_TO_METER_DIV_BY)]
THEIR_CLOSET_BOUNDARY = [(0.25/FEET_TO_METER_DIV_BY, 16.0/FEET_TO_METER_DIV_BY), (2.0/FEET_TO_METER_DIV_BY, 19.0/FEET_TO_METER_DIV_BY)]

# pickup/dropoff locations
OUR_CLOSET_PICKUP = (2.72, 1.025)
OUR_CLOSET_DROPOFF = (3.0, 0.75) # increment y by 0.2 each time we drop off
OUR_ROOM_PICKUP = (2.585, 1.110)
HALLWAY_MOVE = (2.40, 3.05)
OUR_ROOM_DROPOFF = (1.5, 2.20) # decrement y by 0.2 each time we drop off
THEIR_ROOM_PICKUP = (2.142, 5.426)
THEIR_ROOM_DROPOFF = (2.9, 4.1) # increment y by 0.2 each time we drop off
THEIR_CLOSET_PICKUP = (2.142, 5.426)
THEIR_CLOSET_DROPOFF = (1.512, 5.375) # increment y by 0.2 each time we drop off

class MapController():

    # initial conditions
    def __init__(self, ep_robot):
        self.our_position = (1.0, 1.0)
        self.our_rotation = None
        self.their_position = (2.0, 6.0)
        self.ep_robot = ep_robot
        self.ep_chassis = ep_robot.chassis
        self.ep_chassis.sub_attitude(freq=5, callback=self.attitude_callback)
        self.ep_chassis.sub_position(cs=0, freq=5, callback=self.chassis_callback)
        self.frame_rotation = None
        # self.ep_chassis.sub_velocity(freq=5, callback=self.vel_callback)

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
        
        print(f"current position: {self.our_position}")
        print(self.get_current_location())

    def attitude_callback(self, pos):
        yaw, pitch, roll = pos
        if self.frame_rotation is None: # the first time we read the attitude, our yaw should be +90 in the global robot frame (which means it should point in +x in our frame). Create the rotation matrix from this initial angle reading
            theta = np.radians(yaw - 90) # how offset we are from +90
            c, s = np.cos(-theta), np.sin(-theta) # we want the rotation matrix to be the opposite of that angle
            self.frame_rotation = np.array([[c, -s], [s, c]]) # final rotation matrix
            print(self.frame_rotation)
        else:
            theta = np.radians(yaw)
            self.our_rotation = self.frame_rotation @ np.array([[np.cos(theta)], [np.sin(theta)]])
            curr_theta = np.arctan2(self.our_rotation[1], self.our_rotation[0])
            print(f"current rotation: {curr_theta}")

    def vel_callback(self, pos):
        vgx, vgy, vgz, vbx, vby, vbz = pos
        print(f"vgx: {vgx} vgy: {vgy} vgz: {vgz} vbx: {vbx} vby: {vby} vbz: {vbz}")

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

if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH7T001008H")
    map_controller = MapController(ep_robot)
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    ep_led = ep_robot.led
    
    x_vel = 0.0
    y_vel = 0.0
    z_vel = 0.0

    while True:
        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue

        if frame is not None:
            start = time.time()
            if model.predictor:
                model.predictor.args.verbose = False
            result = model.predict(source=frame, show=False)[0]


            # DIY visualization is much faster than show=True for some reason
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().flatten()
                cls = int(box.cls)
                class_label = result.names[cls]
                cv2.rectangle(frame,
                            (int(xyxy[0]), int(xyxy[1])), 
                            (int(xyxy[2]), int(xyxy[3])),
                            color=(0, 0, 255), thickness=2)
                
                cv2.putText(frame, class_label, (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
        
        led_red = clamp(int(abs(x_vel) * 128), 0, 255)
        led_green = clamp(int(abs(y_vel) * 128), 0, 255)
        led_blue = clamp(int(abs(z_vel) * 10), 0, 255)
        map_controller.ep_chassis.drive_speed(x=x_vel, y=y_vel, z=z_vel, timeout=5)
        ep_led.set_led(comp='all', r=led_red, g=led_green, b=led_blue, effect='on')

    map_controller.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
    ep_robot.close()
    exit(0)