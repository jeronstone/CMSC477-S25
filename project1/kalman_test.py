import csv
import numpy as np
from map import *
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
from dijkstra import solve_maze, show_travel

NUM_SUBDIVISIONS = 16

t_spl, x_spl, y_spl = solve_maze()
dt = 0.01
# initialize filter
position_filter = KalmanFilter(dim_x = 4, dim_z = 2)
position_filter.F = np.array([[1, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 0]])
q = Q_discrete_white_noise(dim=2, dt=dt, var=1)
position_filter.Q = block_diag(q, q)
position_filter.H = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0]])
position_filter.R = np.array([[0.00001, 0],
                              [0, 0.00001]])
position_filter.x = np.array([[x_spl(0)*0.2666, 0, y_spl(0)*0.2666, 0]]).T
position_filter.P = np.eye(4) * 0.1
position_filter.B = np.array([[dt, 0], [1, 0], [0, dt], [0, 1]])

x_travel = []
y_travel = []
x_travel_smooth = []
y_travel_smooth = []

with open('traveled_path.csv', newline='') as path_csv:
    path_csv_reader = csv.reader(path_csv, delimiter=',')
    i = 0
    for row in path_csv_reader:
        curr_pt = (float(row[0]), float(row[1]))
        # desired_pt = (x_spl(i/2706.0)*0.2666, y_spl(i/2706.0)*0.2666)
        x_travel.append(float(row[0]))
        y_travel.append(float(row[1]))
        # print(curr_pt)
        # velocity = (desired_pt[0] - curr_pt[0], desired_pt[1] - curr_pt[1])
        # print(velocity)
        # position_filter.predict(u=velocity)
        position_filter.predict(u=None)
        position_filter.update(curr_pt)
        x_travel_smooth.append(position_filter.x[0])
        y_travel_smooth.append(position_filter.x[2])
        # print((position_filter.x[0], position_filter.x[2]))
        i += 1

map1 = Map(os.getcwd() + "\Project1.csv")
fig, ax = plt.subplots(1, 1)
ax.xaxis.set_ticks_position('top')
ax.set_xlim([0, map1.map_num_cols//NUM_SUBDIVISIONS])
ax.set_ylim([map1.map_num_rows//NUM_SUBDIVISIONS, 0])
plt.xticks(np.arange(0, map1.map_num_cols//NUM_SUBDIVISIONS, 1))
plt.yticks(np.arange(0, map1.map_num_rows//NUM_SUBDIVISIONS, 1))
plt.grid('True')
ax.plot(x_spl(t_spl), y_spl(t_spl))
ax.plot([x/0.2666 for x in x_travel], [y/0.2666 for y in y_travel])
ax.plot([x/0.2666 for x in x_travel_smooth], [y/0.2666 for y in y_travel_smooth])
# ax[1].plot(x_spl(t), y_spl(t))
plt.gca().set_aspect('equal')
plt.show()