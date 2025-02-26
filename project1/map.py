from enum import Enum
import csv
import os
from PIL import Image, ImageDraw
from collections import deque

COLOR_FREE = (255, 255, 255)
COLOR_OBSTACLE = (0, 0, 0)
COLOR_START = (0, 255, 0)
COLOR_GOAL = (255, 0, 0)
COLOR_VISITED = (127, 127, 127)
COLOR_PATH = (255, 255, 0)

def get_color(square_type):
    if square_type.value == 0:
        return COLOR_FREE
    if square_type.value == 1:
        return COLOR_OBSTACLE
    if square_type.value == 2:
        return COLOR_START
    if square_type.value == 3:
        return COLOR_GOAL

class MapSquareEnum(Enum):
    FREE = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3

class MapSquare:
    def __init__(self, square_type=MapSquareEnum(0)):
        self.square_type = MapSquareEnum(square_type)
        self.visited = False
        self.in_path = False
        self.parent = (-1, -1)
        self.dist = float('inf')

class Map:
    def __init__(self, map_csv_filepath):
        self.map = [] # 2D array of MapSquares
        # x is the horizontal value (#cols)
        # y is the vertical value (#rows)
        self.start = (-1, -1) # (x,y) of start position
        self.goal = (-1, -1) # (x,y) of goal position
        self.num_rows = -1 # number of rows in map
        self.num_cols = -1 # number of columns in map
        with open(map_csv_filepath, newline='') as map_csv:
            print("Map file loaded from", map_csv_filepath)
            map_csv_reader = csv.reader(map_csv, delimiter=',')
            j = 0
            for row in map_csv_reader:
                map_row = []
                i = 0
                for entry in row:
                    if entry == '':
                        entry = MapSquareEnum.FREE.value
                    map_row.append(MapSquare(square_type=int(entry)))
                    if int(entry) == MapSquareEnum.START.value:
                        self.start = (i, j)
                    if int(entry) == MapSquareEnum.GOAL.value:
                        self.goal = (i, j)
                    i += 1
                self.map.append(map_row)
                j += 1
            self.num_rows = j
            self.num_cols = i
            print("Map initialized. Size =", self.num_rows, "rows x", self.num_cols, "cols. Start =", self.start, ". Goal =", self.goal, ".")

    def visit(self, x, y, parent_x, parent_y):
        self.map[y][x].visited = True
        self.map[y][x].parent = (parent_x, parent_y)

    def draw_path(self):
        path = deque()
        curr_cell = self.goal
        while curr_cell != self.start:
            path.appendleft(curr_cell)
            x = curr_cell[0]
            y = curr_cell[1]
            self.map[y][x].in_path = True
            curr_cell = self.map[y][x].parent
        path.appendleft(curr_cell)
        return path
    
    def gen_img(self, scale=1):
        frame = Image.new("RGB", (self.num_cols * scale, self.num_rows * scale), COLOR_FREE)
        frame_draw = ImageDraw.Draw(frame)
        for cols in range(0, self.num_cols):
            scaled_cols = cols * scale
            for rows in range(0, self.num_rows):
                scaled_rows = rows * scale
                frame_draw.rectangle([(scaled_cols, scaled_rows), (scaled_cols + scale - 1, scaled_rows + scale - 1)], fill=get_color(self.map[rows][cols].square_type))
                if self.map[rows][cols].visited and self.map[rows][cols].square_type is not MapSquareEnum.START and self.map[rows][cols].square_type is not MapSquareEnum.GOAL:
                    frame_draw.rectangle([(scaled_cols, scaled_rows), (scaled_cols + scale - 1, scaled_rows + scale - 1)], fill=COLOR_VISITED)
                if self.map[rows][cols].in_path and self.map[rows][cols].square_type is not MapSquareEnum.START and self.map[rows][cols].square_type is not MapSquareEnum.GOAL:
                    frame_draw.rectangle([(scaled_cols, scaled_rows), (scaled_cols + scale - 1, scaled_rows + scale - 1)], fill=COLOR_PATH)
        return frame

if __name__ == "__main__":
    map1 = Map(os.getcwd() + "\\Maps\\Map1.csv")
    map1.gen_img(scale=10).show()
