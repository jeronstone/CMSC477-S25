# jrana dijkstra.py 2025/02/13

from map import *
from PIL import Image
import os
from collections import deque
import math
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

CROSS_CORNERS = False # whether or not the path can cross an obstacle corner while traversing, since robots would collide with the corner in the real world
ITERATIONS_PER_FRAME = 32 # how many iterations occur before we generate a new frame of the GIF 

def Dijkstra(map, gen_frames, frames, scale=1):
    q = [] # priority queue with priority = dist from source

    # set distance of start to 0 and add it to the queue
    map.getSquareNoScale(*map.start).dist = 0
    q.append((map.start, 0))

    anim_count = 0 # GIF animation control

    while q.count != 0:
        # get current node from priority queue (i.e. node in queue with shortest distance to start) and mark it as visited
        curr_node = min(q, key=lambda p:p[1])
        q.remove(curr_node)
        curr_node = curr_node[0] # get the coordinates
        # curr_x, curr_y = array_to_grid(*curr_node)
        # curr_node = (curr_x, curr_y)
        map.getSquareNoScale(*curr_node).visited = True

        # if this is the goal, return it and end Dijkstra
        if curr_node == map.goal:
            return curr_node

        # check every neighbor, including diagonals
        neighbors = []
        include_neighbors = [[1, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]]
        # top left
        new_x = curr_node[0] - 1
        new_y = curr_node[1] - 1
        top_left = (new_x, new_y)
        if not(new_x >= 0 and new_x < map.map_num_cols and new_y >= 0 and new_y < map.map_num_rows and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE_SOFT):
            include_neighbors[0][0] = 0
        # top middle
        new_x = curr_node[0]
        new_y = curr_node[1] - 1
        top_middle = (new_x, new_y)
        if not(new_x >= 0 and new_x < map.map_num_cols and new_y >= 0 and new_y < map.map_num_rows and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE_SOFT):
            include_neighbors[0][1] = 0
        # if we can't cross corners and this is an obstacle, exclude all neighbors next to this one
            if CROSS_CORNERS is False:
                include_neighbors[0][0] = 0
                include_neighbors[0][1] = 0
                include_neighbors[0][2] = 0
        # top right
        new_x = curr_node[0] + 1
        new_y = curr_node[1] - 1
        top_right = (new_x, new_y)
        if not(new_x >= 0 and new_x < map.map_num_cols and new_y >= 0 and new_y < map.map_num_rows and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE_SOFT):
            include_neighbors[0][2] = 0
        # middle left
        new_x = curr_node[0] - 1
        new_y = curr_node[1]
        middle_left = (new_x, new_y)
        if not(new_x >= 0 and new_x < map.map_num_cols and new_y >= 0 and new_y < map.map_num_rows and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE_SOFT):
            include_neighbors[1][0] = 0
        # if we can't cross corners and this is an obstacle, exclude all neighbors next to this one
            if CROSS_CORNERS is False:
                include_neighbors[0][0] = 0
                include_neighbors[1][0] = 0
                include_neighbors[2][0] = 0
        # middle right
        new_x = curr_node[0] + 1
        new_y = curr_node[1]
        middle_right = (new_x, new_y)
        if not(new_x >= 0 and new_x < map.map_num_cols and new_y >= 0 and new_y < map.map_num_rows and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE_SOFT):
            include_neighbors[1][2] = 0
        # if we can't cross corners and this is an obstacle, exclude all neighbors next to this one
            if CROSS_CORNERS is False:
                include_neighbors[0][2] = 0
                include_neighbors[1][2] = 0
                include_neighbors[2][2] = 0
        # bottom left
        new_x = curr_node[0] - 1
        new_y = curr_node[1] + 1
        bottom_left = (new_x, new_y)
        if not(new_x >= 0 and new_x < map.map_num_cols and new_y >= 0 and new_y < map.map_num_rows and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE_SOFT):
            include_neighbors[2][0] = 0
        # bottom middle
        new_x = curr_node[0]
        new_y = curr_node[1] + 1
        bottom_middle = (new_x, new_y)
        if not(new_x >= 0 and new_x < map.map_num_cols and new_y >= 0 and new_y < map.map_num_rows and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE_SOFT):
            include_neighbors[2][1] = 0
        # if we can't cross corners and this is an obstacle, exclude all neighbors next to this one
            if CROSS_CORNERS is False:
                include_neighbors[2][0] = 0
                include_neighbors[2][1] = 0
                include_neighbors[2][2] = 0
        # bottom right
        new_x = curr_node[0] + 1
        new_y = curr_node[1] + 1
        bottom_right = (new_x, new_y)
        if not(new_x >= 0 and new_x < map.map_num_cols and new_y >= 0 and new_y < map.map_num_rows and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE and map.getSquareNoScale(new_x,new_y).square_type is not MapSquareEnum.OBSTACLE_SOFT):
            include_neighbors[2][2] = 0

        # add included neighbors to list; if second element is 1 then it is a diagonal, otherwise it is not a diagonal
        if include_neighbors[0][0] == 1:
            neighbors.append((top_left, 1))
        if include_neighbors[0][1] == 1:
            neighbors.append((top_middle, 0))
        if include_neighbors[0][2] == 1:
            neighbors.append((top_right, 1))
        if include_neighbors[1][0] == 1:
            neighbors.append((middle_left, 0))
        if include_neighbors[1][2] == 1:
            neighbors.append((middle_right, 0))
        if include_neighbors[2][0] == 1:
            neighbors.append((bottom_left, 1))
        if include_neighbors[2][1] == 1:
            neighbors.append((bottom_middle, 0))
        if include_neighbors[2][2] == 1:
            neighbors.append((bottom_right, 1))
        
        # go through list of neighbors
        for neighbor in neighbors:
            new_dist = map.getSquareNoScale(*curr_node).dist
            # new distance will either be old distance + 1 or old distance + sqrt(2) if diagonal
            if neighbor[1] == 0:
                new_dist += 1.0
            else:
                new_dist += math.sqrt(2)
            neighbor = neighbor[0] # get coordinates
            # if we have found a new minimum distance to this neighbor, update the distance and parent
            if new_dist < map.getSquareNoScale(*neighbor).dist:
                map.getSquareNoScale(*neighbor).parent = curr_node
                map.getSquareNoScale(*neighbor).dist = new_dist
                q.append((neighbor, new_dist))
        
        # GIF animation control
        anim_count += 1
        if gen_frames == True and anim_count >= ITERATIONS_PER_FRAME - 1:
            frames.append(map.gen_img(scale=scale))
            anim_count = 0

def solve_maze(show=False):
    map1 = Map(os.getcwd() + "\Project1.csv")
    Dijkstra(map1, False, [], scale=10)
    path = map1.draw_path()
    x_pts = []
    y_pts = []
    for p in path:
        scaled_x, scaled_y = array_to_grid(*p)
        x_pts.append(scaled_x)
        y_pts.append(scaled_y)
    t = np.linspace(0, 1, num=len(path))
    t_spl = np.linspace(0, 1, num=800)
    x_spl = CubicSpline(t, x_pts)
    y_spl = CubicSpline(t, y_pts)
    
    if show:
        fig, ax = plt.subplots(1, 1)
        ax.xaxis.set_ticks_position('top')
        ax.set_xlim([0, map1.map_num_cols//NUM_SUBDIVISIONS])
        ax.set_ylim([map1.map_num_rows//NUM_SUBDIVISIONS, 0])
        plt.xticks(np.arange(0, map1.map_num_cols//NUM_SUBDIVISIONS, 1))
        plt.yticks(np.arange(0, map1.map_num_rows//NUM_SUBDIVISIONS, 1))
        plt.grid('True')
        ax.plot(x_spl(t_spl), y_spl(t_spl))
        # ax[1].plot(x_spl(t), y_spl(t))
        plt.gca().set_aspect('equal')
        plt.show()
        map1.gen_img(scale=20).show()
    
    return t_spl, x_spl, y_spl

def show_travel(t_spl, x_spl, y_spl, x_trav, y_trav):
    map1 = Map(os.getcwd() + "\Project1.csv")
    fig, ax = plt.subplots(1, 1)
    ax.xaxis.set_ticks_position('top')
    ax.set_xlim([0, map1.map_num_cols//NUM_SUBDIVISIONS])
    ax.set_ylim([map1.map_num_rows//NUM_SUBDIVISIONS, 0])
    plt.xticks(np.arange(0, map1.map_num_cols//NUM_SUBDIVISIONS, 1))
    plt.yticks(np.arange(0, map1.map_num_rows//NUM_SUBDIVISIONS, 1))
    plt.grid('True')
    ax.plot(x_spl(t_spl), y_spl(t_spl))
    ax.plot([x/0.2666 for x in x_trav], [y/0.2666 for y in y_trav])
    # ax[1].plot(x_spl(t), y_spl(t))
    plt.gca().set_aspect('equal')
    plt.show()
    map1.gen_img(scale=20).show()

if __name__ == '__main__':
    solve_maze(show=True)
    # map1 = Map(os.getcwd() + "\Project1.csv")
    # Dijkstra(map1, False, [], scale=10)
    # path = map1.draw_path()
    # map1.gen_img(scale=20).show()