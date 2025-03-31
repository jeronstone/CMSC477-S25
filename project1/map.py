from enum import Enum
import csv
import os
from PIL import Image, ImageDraw
from collections import deque
import numpy as np

NUM_SUBDIVISIONS = 16 # how many squares we will split each CSV grid square into (e.g. value of 2 means each CSV value becomes 2*2=4 map tiles)
OBSTACLE_BUFFER = 0.9 # how many grid spaces away we want to stay from obstacles

def grid_to_array(x: float, y: float):
    return int(x*NUM_SUBDIVISIONS), int(y*NUM_SUBDIVISIONS)

def array_to_grid(x: int, y: int):
    return float(x)/NUM_SUBDIVISIONS, float(y)/NUM_SUBDIVISIONS

COLOR_FREE = (255, 255, 255)
COLOR_VISITED = (127, 127, 127)
COLOR_PATH = (255, 255, 0)
COLOR_OBSTACLE_SOFT = (225, 225, 225)
COLOR_OBSTACLE = (0, 0, 0)
COLOR_START = (0, 255, 0)
COLOR_GOAL = (255, 0, 0)

def get_color(square_type):
    if square_type.value == 0:
        return COLOR_FREE
    if square_type.value == 1:
        return COLOR_VISITED
    if square_type.value == 2:
        return COLOR_PATH
    if square_type.value == 3:
        return COLOR_OBSTACLE_SOFT
    if square_type.value == 4:
        return COLOR_OBSTACLE
    if square_type.value == 5:
        return COLOR_START
    if square_type.value == 6:
        return COLOR_GOAL
    return COLOR_FREE

class MapSquareEnum(Enum):
    FREE = 0
    VISITED = 1
    PATH = 2
    OBSTACLE_SOFT = 3
    OBSTACLE = 4
    START = 5
    GOAL = 6

class MapSquare:
    def __init__(self, square_type=MapSquareEnum(0)):
        self.square_type = MapSquareEnum(square_type)
        self.visited = False
        self.in_path = False
        self.parent = (-1, -1)
        self.dist = float('inf')

    def __repr__(self):
        return str(self.square_type.value)

class Map:
    def __init__(self, map_csv_filepath):
        self.map = np.empty((1, 1), dtype=MapSquare) # 2D array of MapSquares
        # x is the horizontal value (#cols)
        # y is the vertical value (#rows)
        self.start = (-1, -1) # (x,y) of start position
        self.goal = (-1, -1) # (x,y) of goal position
        self.map_num_rows = -1 # number of rows in map
        self.map_num_cols = -1 # number of columns in map

        # open csv
        with open(map_csv_filepath, newline='') as map_csv:
            print("Map file loaded from", map_csv_filepath)
            # count number of rows
            map_csv_reader = csv.reader(map_csv, delimiter=',')
            map_csv_num_rows = sum(1 for row in map_csv_reader)
            # count number of columns
            map_csv.seek(0) # need to reset file position
            map_csv_reader = csv.reader(map_csv, delimiter=',') # need to recreate iterator
            map_csv_num_cols = len(next(map_csv_reader))
            map_csv.seek(0) # need to reset file position
            map_csv_reader = csv.reader(map_csv, delimiter=',') # need to recreate iterator

            # calculate the number of map rows and columns
            self.map_num_rows = map_csv_num_rows*NUM_SUBDIVISIONS + 1
            self.map_num_cols = map_csv_num_cols*NUM_SUBDIVISIONS + 1

            print(f'rows = {self.map_num_rows}, cols = {self.map_num_cols}')

            # create the necessary number of rows and columns in the map matrix
            self.map = np.empty((self.map_num_rows, self.map_num_cols), dtype=MapSquare)

            # begin iterating through CSV
            j = 0
            for row in map_csv_reader:
                i = 0
                for entry in row:
                    if entry == '': # null check: set to free space
                        entry = MapSquareEnum.FREE.value
                    entry_square_value = int(entry)

                    # if this the start or goal, set the object variable and only set the middle square of the subdivision
                    if entry_square_value == MapSquareEnum.START.value:
                        self.start = (i + NUM_SUBDIVISIONS//2, j + NUM_SUBDIVISIONS//2) # center of the split squares
                        self.map[j + NUM_SUBDIVISIONS//2, i + NUM_SUBDIVISIONS//2] = MapSquare(square_type=entry_square_value)
                        entry_square_value = MapSquareEnum.FREE.value # now when we iterate over squares, it will set neighbors to free
                    if entry_square_value == MapSquareEnum.GOAL.value:
                        self.goal = (i + NUM_SUBDIVISIONS//2, j + NUM_SUBDIVISIONS//2) # center of the split squares
                        self.map[j + NUM_SUBDIVISIONS//2, i + NUM_SUBDIVISIONS//2] = MapSquare(square_type=entry_square_value)
                        entry_square_value = MapSquareEnum.FREE.value # now when we iterate over squares, it will set neighbors to free

                    # set map square types based on number of subdivisions
                    for map_y in range(j, j+NUM_SUBDIVISIONS+1):
                        for map_x in range(i, i+NUM_SUBDIVISIONS+1):
                            # map square type has priority equal to its enum
                            if self.map[map_y, map_x] is None:
                                self.map[map_y, map_x] = MapSquare(square_type=entry_square_value)
                            elif self.map[map_y, map_x].square_type.value < entry_square_value:
                                self.map[map_y, map_x].square_type = MapSquareEnum(entry_square_value)
                            
                            # if this is an obstacle, set all neighbors to soft obstacles unless they have a higher priority type
                            if entry_square_value == MapSquareEnum.OBSTACLE.value:
                                scaled_obstacle_buffer = int(OBSTACLE_BUFFER*NUM_SUBDIVISIONS) # what radius around an obstacle we declare as a soft obstacle
                                for new_map_y in range(map_y - scaled_obstacle_buffer, map_y + scaled_obstacle_buffer + 1):
                                    for new_map_x in range(map_x - scaled_obstacle_buffer, map_x + scaled_obstacle_buffer + 1):
                                        if new_map_y == map_y and new_map_x == map_x: # ignore current square
                                            continue
                                        if new_map_y >= 0 and new_map_y < self.map_num_rows and new_map_x >= 0 and new_map_x < self.map_num_cols:
                                            if self.map[new_map_y, new_map_x] is None:
                                                self.map[new_map_y, new_map_x] = MapSquare(square_type=MapSquareEnum.OBSTACLE_SOFT)
                                            elif self.map[new_map_y, new_map_x].square_type.value < MapSquareEnum.OBSTACLE_SOFT.value:
                                                self.map[new_map_y, new_map_x].square_type = MapSquareEnum.OBSTACLE_SOFT
                                

                    i += NUM_SUBDIVISIONS
                j += NUM_SUBDIVISIONS

            scaled_start_x, scaled_start_y = array_to_grid(self.start[0], self.start[1])
            scaled_goal_x, scaled_goal_y = array_to_grid(self.goal[0], self.goal[1])
            print("Map initialized. Size =", self.map_num_rows, "rows x", self.map_num_cols, "cols. Start =", (scaled_start_x, scaled_start_y), ". Goal =", (scaled_goal_x, scaled_goal_y), ".")

    # given a coordinate (x, y) in gridspace (e.g. (2.5, 5.5)), return the corresponding map square from the array (e.g. map[5.5*2, 2.5*2])
    def getSquare(self, x: float, y: float):
        scaled_x, scaled_y = grid_to_array(x, y)
        return self.map[scaled_y, scaled_x]
    
    def getSquareNoScale(self, x: int, y: int):
        return self.map[y][x]

    def draw_path(self):
        path = deque()
        curr_cell = self.goal
        while curr_cell != self.start:
            path.appendleft(curr_cell)
            x = curr_cell[0]
            y = curr_cell[1]
            self.getSquareNoScale(x,y).in_path = True
            curr_cell = self.getSquareNoScale(x,y).parent
        path.appendleft(curr_cell)
        return path
    
    def gen_img(self, scale=1):
        frame = Image.new("RGB", (self.map_num_cols * scale, self.map_num_rows * scale), COLOR_FREE)
        frame_draw = ImageDraw.Draw(frame)
        for cols in range(0, self.map_num_cols):
            scaled_cols = cols * scale
            for rows in range(0, self.map_num_rows):
                scaled_rows = rows * scale
                frame_draw.rectangle([(scaled_cols, scaled_rows), (scaled_cols + scale - 1, scaled_rows + scale - 1)], fill=get_color(self.map[rows][cols].square_type))
                if self.map[rows][cols].visited and self.map[rows][cols].square_type is not MapSquareEnum.START and self.map[rows][cols].square_type is not MapSquareEnum.GOAL:
                    frame_draw.rectangle([(scaled_cols, scaled_rows), (scaled_cols + scale - 1, scaled_rows + scale - 1)], fill=COLOR_VISITED)
                if self.map[rows][cols].in_path and self.map[rows][cols].square_type is not MapSquareEnum.START and self.map[rows][cols].square_type is not MapSquareEnum.GOAL:
                    frame_draw.rectangle([(scaled_cols, scaled_rows), (scaled_cols + scale - 1, scaled_rows + scale - 1)], fill=COLOR_PATH)
        return frame

if __name__ == "__main__":
    map1 = Map(os.getcwd() + "\\Project1.csv")
    print(map1.map)
    map1.gen_img(scale=20).show()
