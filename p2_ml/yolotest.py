from ultralytics import YOLO
import cv2
import time
from robomaster import robot
from robomaster import camera
from ibvs_controller import *
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import sys 

ROBOT_X_VELOCITY_MIN = -0.1
ROBOT_X_VELOCITY_MAX = 0.1
ROBOT_Y_VELOCITY_MIN = -0.15
ROBOT_Y_VELOCITY_MAX = 0.15
ROBOT_Z_VELOCITY_MIN = -1.0
ROBOT_Z_VELOCITY_MAX = 1.0
ROBOT_Z_POSITION_MIN = -50
ROBOT_Z_POSITION_MAX = 50
ROBOT_Z_ANGULAR_VELOCITY_MIN = -0.5
ROBOT_Z_ANGULAR_VELOCITY_MAX = 0.5

robot_z_position = 0 # keeps track of arm position

def clamp(val, min_val, max_val):
    if val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    else:
        return val
    
def sub_data_handler(sub_info):
    pos_x, pos_y = sub_info

    # Fix the integer overflow in pos_y
    if pos_y > 2**31 - 1:
        pos_y = pos_y - 2**32

    # You can use these values to confirm the robot arm is where it is supposed to be
    # It is also usable for determine the right setpoints to send to "moveto" commands
    # print("Robotic Arm: pos x:{0}, pos y:{1}".format(pos_x, pos_y))

print('model')
model = YOLO(r"..\runs_v2\detect\train2\weights\best.pt") # 50 epochs
#model = YOLO(r"C:\Users\jesto\Desktop\CMSC477\CMSC477-S25\runs\detect\train\weights\best.pt") # 25 epochs

# Use vid instead of ep_camera to use your laptop's webcam
# vid = cv2.VideoCapture(0)


ep_robot = robot.Robot()
ep_robot.initialize(conn_type="sta", sn="3JKCH7T001008H")
ep_chassis = ep_robot.chassis
ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
ep_arm = ep_robot.robotic_arm
ep_gripper = ep_robot.gripper

ep_gripper.open(power=100)
time.sleep(2.0)
ep_gripper.pause()

ep_gripper.close(power=100)
time.sleep(2.0)
ep_gripper.pause()
ep_arm.moveto(x=200, y=50).wait_for_completed()


# Start printing the gripper position
# ep_arm.sub_position(freq=5, callback=sub_data_handler)

controller = IBVS_Controller(control_mode='4xyzy', interaction_mode='desired', num_pts=4)
controller.set_lambda_matrix([2.0, 10.0, 1.0, 1000.0]) # robot y velocity; robot z velocity; robot x velocity; robot z angular velocity
controller.set_desired_points([(-0.16, 0.2, 0.215), (0.16, 0.2, 0.215), (-0.16, 0.9, 0.215), (0.16, 0.9, 0.215)])

while True:
    # ret, frame = vid.read()
    try:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
    except:
        frame = None
        continue
    if frame is not None:
        start = time.time()
        if model.predictor:
            model.predictor.args.verbose = False
            
        #print('before')
        result = model.predict(source=frame, show=False)[0]
        boxes = result.boxes
        #print('after')
        
        #print(result)
                
        # for r in result:
        #     for b in r.boxes:
        #         if int(b.cls) != 1:
        #             continue
        #         xy = r.masks.xy  # mask in polygon format
        #         xyn = r.masks.xyn  # normalized
        #         masks = r.masks.data  # mask in matrix format (num_objects x H x W)
                
        #         if len(masks.numpy()) < 2:
        #             continue
                
        #         msk = masks.numpy()[1]
        #         mask = np.uint8(msk)
                        
        #         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #         epsilon = 0.02 * cv2.arcLength(contours[0], True)  # epsilon controls the approximation accuracy
        #         approx = cv2.approxPolyDP(contours[0], epsilon, True)
                
        #         for corner in approx:
        #             cv2.circle(frame, tuple(corner[0]), 5, (0, 255, 0), -1)

        #         for point in xy[1]:
        #             x, y = point.astype(int)
        #             cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Red dot
        #         continue
        #         points = xy[1]
        #         hull = ConvexHull(points)
        #         plt.plot(points[:,0], points[:,1], 'o')
        #         for simplex in hull.simplices:
        #             plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        #         plt.show()
        # #names = model.names

        # # DIY visualization is much faster than show=True for some reason
        # #boxes = result.boxes
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)  # Small wait so the window updates
        # continue
        
        if len(boxes) == 0:
            # ep_chassis.drive_speed(x=0, y=0, z=-10, timeout=5)
            pass
        else:
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().flatten()
                cv2.rectangle(frame,
                            (int(xyxy[0]), int(xyxy[1])), 
                            (int(xyxy[2]), int(xyxy[3])),
                            color=(0, 0, 255), thickness=2)
                
                corners = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                corners[0] = 2*((corners[0]) / 640) - 1
                corners[2] = 2*((corners[2]) / 640) - 1
                
                corners[1] = 2*((corners[1]) / 360) - 1
                corners[3] = 2*((corners[3]) / 360) - 1

                cv2.putText(frame, str((round(corners[2], 2), round(corners[1], 2))), (int(xyxy[2]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
                cv2.putText(frame, str((round(corners[0], 2), round(corners[3], 2))), (int(xyxy[0]), int(xyxy[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
                cv2.putText(frame, str((round(corners[2], 2), round(corners[3], 2))), (int(xyxy[2]), int(xyxy[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
                cv2.putText(frame, str((round(corners[0], 2), round(corners[1], 2))), (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
                
                continue
                # print(corners)
                controller.set_current_points([(corners[0], corners[1], None), (corners[2], corners[1], None), (corners[0], corners[3], None), (corners[2], corners[3], None)])
                controller.calculate_interaction_matrix()
                vels = controller.calculate_velocities()
                # print(f"vels: {vels}")

                # robot x velocity is camera z velocity
                robot_x_velocity = vels[2][0]
                #print(robot_x_velocity)
                robot_x_velocity = clamp(robot_x_velocity, ROBOT_X_VELOCITY_MIN, ROBOT_X_VELOCITY_MAX)

                # robot y velocity is camera x velocity
                robot_y_velocity = vels[0][0]
                robot_y_velocity = clamp(robot_y_velocity, ROBOT_Y_VELOCITY_MIN, ROBOT_Y_VELOCITY_MAX)

                # robot z velocity is inverted camera y velocity
                robot_z_velocity = -vels[1][0]
                robot_z_velocity = clamp(robot_z_velocity, ROBOT_Z_VELOCITY_MIN, ROBOT_Z_VELOCITY_MAX)
                robot_z_position += robot_z_velocity
                robot_z_position = clamp(robot_z_position, ROBOT_Z_POSITION_MIN, ROBOT_Z_POSITION_MAX)

                # robot z angular velocity is camera y angular velocity
                robot_z_angular_velocity = vels[3][0]
                robot_z_angular_velocity = clamp(robot_z_angular_velocity, ROBOT_Z_ANGULAR_VELOCITY_MIN, ROBOT_Z_ANGULAR_VELOCITY_MAX)

                # send robot z position to arm
                # ep_arm.moveto(x=200, y=robot_z_position).wait_for_completed()

                # send robot x, y, and angular z velocities to robot
                # ep_chassis.drive_speed(x=robot_x_velocity, y=robot_y_velocity, z=robot_z_angular_velocity, timeout=5)

                #print(f"vels: x {robot_x_velocity} y {robot_y_velocity} z {robot_z_velocity} z ang {robot_z_angular_velocity}; arm pos: {robot_z_position}")


        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # print(results)


        end = time.time()
        #print(1.0 / (end-start))

ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
ep_robot.close()