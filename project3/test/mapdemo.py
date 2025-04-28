import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure and axes
fig, ax = plt.subplots(1)

# Define the bottom-left corner, width, and height of the rectangle
x, y = 1, 2
width = 5
height = 3

OUR_CLOSET_BOUNDARY = [(2.6, 0.60), (3.6, 1.5)]
CLOSET_H = 1
CLOSET_W = 0.9
OUR_ROOM_BOUNDARY = [(0.25, 0.25), (2.6, 2.25)]
ROOM_H = 2
ROOM_W = 2.35
HALLWAY_BOUNDARY = [(1.0, 2.5), (2.6, 4.5)]
HALL_H = 2
HALL_W = 1.6

# Create a Rectangle patch
rect = patches.Rectangle(OUR_CLOSET_BOUNDARY[0], CLOSET_W, CLOSET_H, linewidth=1, edgecolor='r', facecolor='none')
rect1 = patches.Rectangle(OUR_ROOM_BOUNDARY[0], ROOM_W, ROOM_H, linewidth=1, edgecolor='g', facecolor='none')
rect2 = patches.Rectangle(HALLWAY_BOUNDARY[0], HALL_W, HALL_H, linewidth=1, edgecolor='b', facecolor='none')

# Add the patch to the axes
ax.add_patch(rect)
ax.add_patch(rect1)
ax.add_patch(rect2)

# Set the limits of the plot to accommodate the rectangle
ax.set_xlim(0, 4)
ax.set_ylim(0, 5)

# Show the plot
plt.show()