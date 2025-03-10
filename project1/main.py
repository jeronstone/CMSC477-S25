import cv2
import numpy as np
import time
import traceback
from queue import Empty
from robomaster import robot
from robomaster import camera
from ApriltagDetector import *
from ibvs import get_ibvs_speeds
from dijkstra import solve_maze, show_travel
import csv
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

APRILTAG_SIZE = 0.2666
USE_IBVS = False

april_left  = np.array([[0,0,1],[1,0,0],[0,1,0]]) #np.array([[1,0,0],[0,0,1],[0,-1,0]])   # -90deg x rotation
april_right = np.array([[0,0,-1],[-1,0,0],[0,1,0]]) #np.array([[-1,0,0],[0,0,-1],[0,-1,0]]) # -90deg x rotation, 180deg z rotation
april_up    = np.array([[-1,0,0],[0,0,1],[0,1,0]]) #np.array([[0,0,1],[0,1,0],[-1,0,0]])   # 90deg y rotation
april_down  = np.array([[1,0,0],[0,0,-1],[0,1,0]]) #np.array([[0,0,-1],[1,0,0],[0,-1,0]])  # -90deg y rotation, 180deg z rotation
april_to_coords = {
    30: (4.0,   1.5, april_left),
    31: (5.0,   1.5, april_right),
    32: (4.0,   3.5, april_left),
    33: (5.0,   3.5, april_right),
    34: (4.5,   5.0, april_down),  
    35: (6.5,   1.0, april_down),
    36: (8.5,   1.0, april_down),    
    37: (7.5,   4.0, april_up),  
    38: (7.0,   5.5, april_left),
    39: (8.0,   5.5, april_right),
    40: (7.0,   7.5, april_left),
    41: (8.0,   7.5, april_right),
    42: (10.0,  1.5, april_left),
    43: (11.0,  1.5, april_right),
    44: (10.0,  3.5, april_left),
    45: (11.0,  3.5, april_right),
    46: (10.5,  5.0, april_down)
}

def draw_detections(frame, detections, coords=None):
    for detection in detections:
        if detection.tag_id < 30 or detection.tag_id > 46:
            continue
        pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)

        frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        top_left = tuple(pts[0][0])  # First corner
        top_right = tuple(pts[1][0])  # Second corner
        bottom_right = tuple(pts[2][0])  # Third corner
        bottom_left = tuple(pts[3][0])  # Fourth corner
        cv2.line(frame, top_left, bottom_right, color=(0, 0, 255), thickness=2)
        cv2.line(frame, top_right, bottom_left, color=(0, 0, 255), thickness=2)
        center_x = int(((top_left[0] + top_right[0])/2) - 4*len(coords[detection.tag_id]))
        center_y = int((top_left[1] + bottom_left[1])/2)
        cv2.putText(frame, str(detection.tag_id) + ":" + coords[detection.tag_id], (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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
    e_threshold = 0.065

    first_detect = 0
    prev_t = time.time()
    dt = 0.1
    prev_error = np.array([0.0, 0.0])
    integral = np.array([0.0, 0.0])
    closest_april_id = -1
    coords = {}
    prev_velos = None

    x_travel = []
    y_travel = []

    # use initial apriltag to find current position in world frame
    initial_tag = 32

    f = open('traveled_path2.csv','w')
    writer = csv.writer(f)

    # initialize filter
    # position_filter = KalmanFilter(dim_x = 4, dim_z = 2)
    # position_filter.F = np.array([[1, 0, 0, 0],
    #                               [0, 0, 0, 0],
    #                               [0, 0, 1, 0],
    #                               [0, 0, 0, 0]])
    # q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
    # position_filter.Q = block_diag(q, q)
    # position_filter.H = np.array([[1, 0, 0, 0],
    #                               [0, 0, 1, 0]])
    # position_filter.R = np.array([[1, 0],
    #                               [0, 1]])
    # position_filter.x = np.array([[x_spl(0), 0, y_spl(0), 0]]).T
    # position_filter.P = np.eye(4) * 5
    # position_filter.B = np.array([[dt, 0], [1, 0], [0, dt], [0, 1]])
    # position_filter.update((x_spl(3.0/800), y_spl(3.0/800)))
    # print(f'update: {position_filter.x}')
    
    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue

        curr_t = time.time()
        dt = curr_t - prev_t

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)

        detections = apriltag.find_tags(gray)
        pe_wc = []

        if len(detections) > 0:
            closest_T_wc = None
            closest_tca = None
            minscore = float('inf')
            for detection in detections:
                if detection.tag_id < 30 or detection.tag_id > 46:
                    continue
                #if detection.tag_id == initial_tag:
                # create T_wa matrix (from world frame to apriltag frame)
                tag_position = april_to_coords[detection.tag_id]
                T_wa = np.array([[tag_position[2][0,0], tag_position[2][0,1], tag_position[2][0,2], tag_position[0]*APRILTAG_SIZE], 
                                 [tag_position[2][1,0], tag_position[2][1,1], tag_position[2][1,2], tag_position[1]*APRILTAG_SIZE],
                                 [tag_position[2][2,0], tag_position[2][2,1], tag_position[2][2,2],            -0.5*APRILTAG_SIZE],
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
                coords[detection.tag_id] = str((round(T_wc[0,3]/APRILTAG_SIZE,3), round(T_wc[1,3]/APRILTAG_SIZE,3)))
                pe_wc.append((T_wc[0,3], T_wc[1,3]))
                # print(((tag_position[0]*0.266 + t_ca[2])/0.266, (tag_position[1]*0.266 + t_ca[0])/0.266))
                # print(T_ac)

                closeness_score = 0.3*np.linalg.norm(t_ca-np.array([0, 0, APRILTAG_SIZE]))
                rotation_score = 0.7*np.linalg.norm((np.identity(3)-R_ca))

                final_score = closeness_score+rotation_score

                if (closest_tca is None or final_score < minscore):# and t_ca[2] < APRILTAG_SIZE*3.5:#np.linalg.norm(np.array([t_ca[0], t_ca[1], t_ca[2]])) < np.linalg.norm(np.array([closest_tca[0], closest_tca[1], closest_tca[2]])):# and t_ca[2] < APRILTAG_SIZE*2:
                    closest_tca = t_ca
                    closest_T_wc = T_wc #(T_wc[1,3], T_wc[0,3])
                    closest_april_id = detection.tag_id
                    minscore = final_score

                #break
            
            if closest_T_wc is None:
                #ep_chassis.drive_speed(x=0, y=0, z=-20, timeout=5) # spin in place
                continue

            avg_pos = [0, 0]
            # for pe_i in pe_wc:
            #     for j in range(2):
            #         avg_pos[j] += (pe_i[j]/len(pe_wc))
            #         # optional: weight by |t_ca| proximity to 0 (close to middle of camera frame)

            #print(closest_T_wc)
            avg_pos = (closest_T_wc[0,3], closest_T_wc[1,3])
            T_cw = np.linalg.inv(closest_T_wc)
            R_cw = np.array([[T_cw[0,0], T_cw[0,1], T_cw[0,2]],
                            [T_cw[1,0], T_cw[1,1], T_cw[1,2]],
                            [T_cw[2,0], T_cw[2,1], T_cw[2,2]]])
            
            # position_filter.F = np.array([[1, dt, 0,  0],
            #                               [0,  1, 0,  0],
            #                               [0,  0, 1, dt],
            #                               [0,  0, 0,  1]])
            # if prev_velos is not None:
            #     position_filter.predict(u=prev_velos)
            # position_filter.update(avg_pos)
            # avg_pos = np.array((position_filter.x[0,0], position_filter.x[2,0]))
            
            avg_pos = np.array(avg_pos)
            pd_w_t = np.array([x_spl(t/len(t_spl))*0.266, y_spl(t/len(t_spl))*0.266])

            #print(f'BEFORE: \t curr_pos: {avg_pos} \t spl: {pd_w_t}')

            writer.writerow(avg_pos)
            x_travel.append(avg_pos[0])
            y_travel.append(avg_pos[1])

            e_w = avg_pos - pd_w_t
            #e_w = avg_pos - np.array([9.0*0.266, 7.5*0.266])

            if np.linalg.norm(e_w) < e_threshold:
                if (t < len(t_spl)):
                    t+=1.0
                    integral=np.array([0,0])
                else:
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
                    show_travel(t_spl, x_spl, y_spl, x_travel, y_travel)
                    break
            
            #ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)

            
            
            integral = integral+(e_w*dt)
            deriv = (e_w - prev_error) / dt

            if USE_IBVS:
                # lam = np.matrix([[-1.0, 0.0], [0.0, -1.5]])

                # x = closest_tca[0] / closest_tca[2]
                # y = closest_tca[1] / closest_tca[2]
                # Lx = np.matrix([[x/closest_tca[2], -1.0/closest_tca[2]], [y/closest_tca[2], 0.0]])

                # e_c = np.matmul(R_cw, np.array([e_w[0], e_w[1], 0]).T)
                
                # res = lam @ Lx.I @ np.array([[float(e_c[2])], [float(e_c[0])]])
                # # R_bc = np.array([[0,0,1],[1,0,0],[0,-1,0]])
                # # res = R_bc @ res

                # print(f'{res}')

                lam = np.matrix([[0.5, 0.0], [0.0, 0.75]])

                x = closest_tca[0] / closest_tca[2]
                y = closest_tca[1] / closest_tca[2]
                Lx = np.matrix([[-1.0/closest_tca[2],0.0], [0.0,-1.0/closest_tca[2]]])

                e_c = np.matmul(R_cw, np.array([e_w[0], e_w[1], 0]).T)
                R_bc = np.array([[0,0,1],[1,0,0],[0,-1,0]])
                e_b = np.matmul(R_bc, e_c)

                res = lam @ Lx.I @ np.array([[float(e_b[0])], [float(e_b[1])]])
                print(f'{res}')

                # x = closest_tca[0] / closest_tca[2]
                # y = closest_tca[1] / closest_tca[2]

                # Lx = np.matrix([[x,-1*(1+x**2)], [y,-1*x*y]])

                # lam = np.matrix([[-75, 0], [0, -3]])

                #print(f'e_c: {np.array([[-e_c[2]], [-e_c[0]]])} \n Lx: {Lx.I} \n lam: {lam}')

                #res = np.matmul(lam, np.matmul(Lx.I, np.array([[-e_c[2]], [-e_c[0]]])))

                #print(f'{res.item(1)} \t {res.item(0)}')

                velos = [res.item(0), res.item(1)]
                #velos = [0,0,0]
                #print(f'curr_pos: {avg_pos} \t from tag {closest_april_id} \t  spl: {pd_w_t} \t t: {t} Velos: {velos}')

                ep_chassis.drive_speed(x=velos[0], y=velos[1], z=15*closest_tca[0], timeout=5)
            else:
                velos = [0, 0]
                velos[0] = -1.0*e_w[0] -0.1*integral[0] -0.5*deriv[0]
                velos[1] = -1.0*e_w[1] -0.1*integral[1] -0.5*deriv[1]

                velos = np.matmul(R_cw, np.array([velos[0], velos[1], 0]).T)
                print(f'curr_pos: {avg_pos} \t from tag {closest_april_id} scored {round(minscore, 2)} \t deriv: {deriv} \t  spl: {pd_w_t} \t t: {t} \t Err: {e_w} \t Velos: {velos}')
                ep_chassis.drive_speed(x=velos[2], y=velos[0], z=15*closest_tca[0], timeout=5)
                prev_velos = [velos[2], velos[0]]

            #print(f'BEFORE: \t curr_pos: {avg_pos} \t spl: {pd_w_t} \t t: {t} \t Err: {e_w} \t Velos: {velos}')

            #print(R_cw)
            
            #print(f'curr_pos: {avg_pos} \t from tag {closest_april_id} \t  spl: {pd_w_t} \t t: {t} \t Err: {e_w} \t Velos: {velos}')
            #print(f'AFTER : \t curr_pos: {avg_pos} \t spl: {pd_w_t} \t t: {t} \t Err: {e_w} \t Velox: {velos[1]} \t Veloy: {-velos[0]}')

            #ep_chassis.drive_speed(x=velos[2], y=velos[0], z=15*closest_tca[0], timeout=5)
            #ep_chassis.drive_speed(x=velos[2], y=velos[0], z=0, timeout=5)
            #ep_chassis.drive_speed(x=-velos[1], y=velos[0], z=0, timeout=5)

            prev_error = e_w
            prev_t = curr_t

        else:
            ep_chassis.drive_speed(x=0, y=0, z=-20, timeout=5) # spin in place

        draw_detections(img, detections, coords)
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
            show_travel(t_spl, x_spl, y_spl, x_travel, y_travel)
            f.close()
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
