import numpy as np
from ApriltagDetector import get_pose_apriltag_in_camera_frame

# @param detection is one element from list returned by ApriltagDetector.find_tags()
def get_ibvs_speeds(detection, offset):

    t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
    print('t_ca', t_ca)
    #print('R_ca', R_ca)

    x = t_ca[0] / t_ca[2]
    y = t_ca[1] / t_ca[2]

    Lx = np.matrix([[x,-1*(1+x**2)], [y,-1*x*y]])

    deltas = np.matrix([[(t_ca[2] - offset[0])], [(t_ca[0]) - offset[1]]])

    #lam = np.matrix([[-75, 0], [0, -3]])
    lam = np.matrix([[-3, 0], [0, -1]])

    res = np.matmul(lam, np.matmul(Lx.I, deltas))

    #print(f'{res.item(1)} \t {res.item(0)}')
    return res, deltas