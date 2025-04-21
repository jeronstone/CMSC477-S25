from robomaster import robot
from robomaster import camera
import time
import cv2

def sp_callback(pos):
    x,y,z = pos
    print(f'{x},{y},{z}')

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH7T001008H")

    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    
    ep_chassis.sub_position(cs=0, freq=5, callback=sp_callback)

    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    
    print('init')

    while True:
        # ret, frame = vid.read()
        frame = ep_camera.read_cv2_image()
        cv2.imshow("Camera Feed", frame)
        if frame is not None:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('w'):
                ep_chassis.drive_speed(x=0.5, y=0, z=0, timeout=5)
            elif key == ord('a'):
                ep_chassis.drive_speed(x=0, y=-0.5, z=0, timeout=5)
            elif key == ord('s'):
                ep_chassis.drive_speed(x=-0.5, y=0, z=0, timeout=5)
            elif key == ord('d'):
                ep_chassis.drive_speed(x=0, y=0.5, z=0, timeout=5)
            elif key == ord('e'):
                ep_chassis.drive_speed(x=0, y=0, z=10, timeout=5)
            elif key == ord('r'):
                ep_chassis.drive_speed(x=0, y=0, z=-10, timeout=5)
            else:
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
    
    
    ep_camera.stop_video_stream()
    ep_chassis.unsub_position()
    ep_robot.close()
