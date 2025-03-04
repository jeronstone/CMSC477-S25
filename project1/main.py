import cv2
import numpy as np
import time
import traceback
from queue import Empty
from robomaster import robot
from robomaster import camera
from ApriltagDetector import *
from ibvs import get_ibvs_speeds
from dijkstra import solve_maze

APRILTAG_SIZE = 0.2666

april_left  = np.array([[1,0,0],[0,0,1],[0,-1,0]])   # -90deg x rotation
april_right = np.array([[-1,0,0],[0,0,-1],[0,-1,0]]) # -90deg x rotation, 180deg z rotation
april_up    = np.array([[0,0,1],[0,1,0],[-1,0,0]])   # 90deg y rotation
april_down  = np.array([[0,0,-1],[1,0,0],[0,-1,0]])  # -90deg y rotation, 180deg z rotation
april_to_coords = {
    30: (3.0,   2.5, april_left),
    31: (4.0,   2.5, april_right),
    32: (3.0,   4.5, april_left),
    33: (4.0,   4.5, april_right),
    34: (3.5,   6.0, april_down),  
    35: (5.5,   2.0, april_down),
    36: (7.5,   2.0, april_down),    
    37: (6.5,   5.0, april_up),  
    38: (6.0,   6.5, april_left),
    39: (7.0,   6.5, april_right),
    40: (6.0,   8.5, april_left),
    41: (7.0,   8.5, april_right),
    42: (9.0,   2.5, april_left),
    43: (10.0,  2.5, april_right),
    44: (9.0,   4.5, april_left),
    45: (10.0,  4.5, april_right),
    46: (9.5,   6.0, april_down)
}

def draw_detections(frame, detections, coords=None):
    for detection in detections:
        pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)

        frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        top_left = tuple(pts[0][0])  # First corner
        top_right = tuple(pts[1][0])  # Second corner
        bottom_right = tuple(pts[2][0])  # Third corner
        bottom_left = tuple(pts[3][0])  # Fourth corner
        cv2.line(frame, top_left, bottom_right, color=(0, 0, 255), thickness=2)
        cv2.line(frame, top_right, bottom_left, color=(0, 0, 255), thickness=2)
        #center_x = int(((top_left[0] + top_right[0])/2) - 4*len(coords[detection.tag_id]))
        #center_y = int((top_left[1] + bottom_left[1])/2)
        #cv2.putText(frame, str(detection.tag_id) + ":" + coords[detection.tag_id], (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

if __name__ == '__main__':
    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100YR")
    ep_chassis = ep_robot.chassis
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel
    marker_size_m = 0.153 # Size of the AprilTag in meters
    apriltag = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)

    t_spl, x_spl, y_spl = solve_maze()
    t = 0.0
    e_threshold = 0.075

    first_detect = 0
    prev_t = time.time()
    prev_error = 0
    integral = 0

    # use initial apriltag to find current position in world frame
    initial_tag = 43

    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)

        detections = apriltag.find_tags(gray)
        pe_wc = []

        if len(detections) > 0:
            closest_T_wc = None
            prev_tca = None
            for detection in detections:
                #if detection.tag_id == initial_tag:
                # create T_wa matrix (from world frame to apriltag frame)
                tag_position = april_to_coords[detection.tag_id]
                T_wa = np.array([[tag_position[2][0,0], tag_position[2][0,1], tag_position[2][0,2], tag_position[1]*APRILTAG_SIZE], 
                                 [tag_position[2][1,0], tag_position[2][1,1], tag_position[2][1,2], tag_position[0]*APRILTAG_SIZE],
                                 [tag_position[2][2,0], tag_position[2][2,1], tag_position[2][2,2],             0.5*APRILTAG_SIZE],
                                 [                   0,                    0,                    0,                             1]])
                
                # create T_ac matrix (from apriltag frame to camera frame)
                t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
                # y_180_rot = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
                # R_ca = np.matmul(R_ca, y_180_rot)
                T_ca = np.array([[R_ca[0,0], R_ca[0,1], R_ca[0,2], t_ca[0]], 
                                 [R_ca[1,0], R_ca[1,1], R_ca[1,2], t_ca[1]],
                                 [R_ca[2,0], R_ca[2,1], R_ca[2,2], t_ca[2]],
                                 [        0,         0,         0,       1]])
                T_ac = np.linalg.inv(T_ca)

                # multiply them to get the position of the camera in the world
                T_wc = np.matmul(T_wa, T_ac)
                # print(T_wc)
                #print((T_wc[1,3]/APRILTAG_SIZE, T_wc[0,3]/APRILTAG_SIZE, T_wc[2,3]/APRILTAG_SIZE))
                pe_wc.append((T_wc[0,3], T_wc[1,3]))
                # print(((tag_position[0]*0.266 + t_ca[2])/0.266, (tag_position[1]*0.266 + t_ca[0])/0.266))
                # print(T_ac)
                # print(T_ac)

                if prev_tca is None or np.linalg.norm(np.array([t_ca[0], t_ca[1]])) < np.linalg.norm(np.array([prev_tca[0], prev_tca[1]])):
                    prev_tca = t_ca
                    closest_T_wc = T_wc #(T_wc[1,3], T_wc[0,3])

                #break
            
            # avg_pos = [0, 0]
            # for pe_i in pe_wc:
            #     for j in range(2):
            #         avg_pos[j] += (pe_i[j]/len(pe_wc))
            #         # optional: weight by |t_ca| proximity to 0 (close to middle of camera frame)

            avg_pos = (closest_T_wc[0,3], closest_T_wc[1,3])
            T_cw = np.linalg.inv(closest_T_wc)
            R_cw = np.array([[T_cw[0,0], T_cw[0,1], T_cw[0,2]],
                            [T_cw[1,0], T_cw[1,1], T_cw[1,2]],
                            [T_cw[2,0], T_cw[2,1], T_cw[2,2]]]) 
            
            avg_pos = np.array(avg_pos)
            pd_w_t = np.array([y_spl(t/len(t_spl))*0.266, x_spl(t/len(t_spl))*0.266])

            #print(f'BEFORE: \t curr_pos: {avg_pos} \t spl: {pd_w_t}')

            e_w = avg_pos - pd_w_t

            if np.linalg.norm(e_w) < e_threshold:
                t+=1.0
                if (t > len(t_spl)):
                    break
            
            #ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)

            # curr_t = time.time()
            # dt = curr_t - prev_t
            
            # integral += (e_w*dt)
            # deriv = (e_w - prev_error) / dt

            velos = [0, 0]
            velos[1] = -0.75*e_w[1] #+ -0.1*integral #+ 100.0*deriv
            velos[0] = -5*e_w[0]

            print(f'BEFORE: \t curr_pos: {avg_pos} \t spl: {pd_w_t} \t t: {t} \t Err: {e_w} \t Velos: {velos}')

            velos = np.matmul(R_cw, np.array([[velos[1]], [velos[0]], [0]]))
            #print(R_cw)
            
            print(f'AFTER : \t curr_pos: {avg_pos} \t spl: {pd_w_t} \t t: {t} \t Err: {e_w} \t Velox: {-velos[0,0]} \t Veloy: {velos[1,0]}')

            ep_chassis.drive_speed(x=-velos[0, 0], y=velos[1, 0], z=0, timeout=5)

            # prev_error = e_w
            # prev_t = curr_t

        else:
            ep_chassis.drive_speed(x=0, y=0, z=-15, timeout=5) # spin in place

        draw_detections(img, detections)
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
            break

    ### TRAJECTORY FOLLOWING PSEUDOCODE:
    # start program, init robot, etc.
    # start capturing images
    # for each capture:
        # detect apriltags in capture
        # if no detections:
            # spin in place
        # else:
            # for each detection:
                # get camera pose estimation in world frame: pe_wc_i
                # (optional: store how close the apriltag is to the center of the image)
            # get the average of all estimations: 1/n * sum(pe_wc_i) over all i: pe_wc
            # (optional: make it a weighted average based on how close the apriltag is to the center of the image, since there are distortions at the camera edges)
            # if |pe_wc - pd_wc(t)| < threshold, where pd_wc(t) is desired pose at time t:
                # t += 1
                # if t > path length:
                    # stop motors
                    # exit capture loop
            # error = pe_wc - pd_wc(t)
            # PID calculations; use error and dt to set motor speed (also stop spinning by setting angular to 0)
            # PID calculations; use error and dt to set motor speed
                

        


    # offsets = [
    #     (33, (1, 0)), 
    #     (33, (2, 0)),
    #     (33, (2, 1))
    # ]
    # curr = 0

    # while (True):
    #     try:
    #         img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
    #     except Empty:
    #         time.sleep(0.001)
    #         continue

    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     gray.astype(np.uint8)

    #     detections = apriltag.find_tags(gray)

    #     if curr < len(offsets):

    #         if len(detections) < 1:
    #             ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)

    #         for detect in detections:
    #             if detect.tag_id == offsets[curr][0]:
    #                 res, deltas = get_ibvs_speeds(detect, offsets[curr][1])

    #                 error = np.linalg.norm(deltas)
    #                 #print(f'Error: {error}')

    #                 if error > 0.05:
    #                     ep_chassis.drive_speed(x=res.item(1), y=0, z=res.item(0), timeout=5)
    #                 else:
    #                     ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
    #                     curr += 1
    #                     print('curr ++')
    #                     break

    #     else:
    #         ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
        
    #     draw_detections(img, detections)
        # cv2.imshow("img", img)
        # if cv2.waitKey(1) == ord('q'):
        #     ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
        #     break
