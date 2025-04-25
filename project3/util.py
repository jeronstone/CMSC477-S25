from enum import Enum

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
    
# callback for chassis subposition
def chassis_subpos_cb(pos):
    x,y,z = pos
    return -x, y, z

# action enum
class Action(Enum):
    PICKUP_BLOCK_2x2 = 0,
    PICKUP_BLOCK_2x4 = 1,
    PICKUP_BLOCK_4x4 = 2,
    DROP_BLOCK = 3,
    MOVE_OUR_CLOSET = 10,
    MOVE_OUR_ROOM = 11,
    MOVE_HALLWAY = 12,
    MOVE_THEIR_ROOM = 13,
    MOVE_THEIR_CLOSET = 14,