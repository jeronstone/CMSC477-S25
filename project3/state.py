from util import *

class State():
        
    def __init__(self):

        # initial conditions
        self.our_position = "OUR_ROOM"
        self.their_position = "THEIR_ROOM"
        self.our_held_block = None
        self.their_held_block = None
        #self.block_positions = [("2x2", "OUR_CLOSET"), ("2x4", "OUR_CLOSET"), ("4x4", "OUR_CLOSET"), ("2x2", "THEIR_CLOSET"), ("2x4", "THEIR_CLOSET"), ("4x4", "THEIR_CLOSET")]
        self.block_positions = [("4x4", "OUR_CLOSET"), ("4x4", "OUR_CLOSET"), ("4x4", "OUR_CLOSET"), ("4x4", "THEIR_CLOSET"), ("4x4", "THEIR_CLOSET"), ("4x4", "THEIR_CLOSET")]
        self.current_score = None
        self.update_and_get_score()

    def copy(self):
        new_state = State()
        new_state.our_position = self.our_position
        new_state.their_position = self.their_position
        new_state.our_held_block = self.our_held_block
        new_state.their_held_block = self.their_held_block
        new_state.block_positions = self.block_positions.copy()
        return new_state

    def __repr__(self):
        return f"STATE:\
                \n\tOUR POS: {self.our_position}\
                \n\tTHEIR POS: {self.their_position}\
                \n\tOUR HELD BLOCK: {self.our_held_block}\
                \n\tTHEIR HELD BLOCK: {self.their_held_block}\
                \n\tBLOCK POSITIONS: {self.block_positions}\
                \n\tCURRENT SCORE: {self.current_score}"

    def update_and_get_score(self):
        
        score_change = 0
        
        for bp in self.block_positions:
            if bp[1] == "OUR_CLOSET":
                if bp[0] == "2x2":
                    score_change -= 16
                elif bp[0] == "2x4":
                    score_change -= 8
                elif bp[0] == "4x4":
                    score_change -= 4
            elif bp[1] == "OUR_ROOM":
                if bp[0] == "2x2":
                    score_change -= 4
                elif bp[0] == "2x4":
                    score_change -= 2
                elif bp[0] == "4x4":
                    score_change -= 1
            elif bp[1] == "THEIR_ROOM":
                if bp[0] == "2x2":
                    score_change += 4
                elif bp[0] == "2x4":
                    score_change += 2
                elif bp[0] == "4x4":
                    score_change += 1
            elif bp[1] == "THEIR_CLOSET":
                if bp[0] == "2x2":
                    score_change += 16
                elif bp[0] == "2x4":
                    score_change += 8
                elif bp[0] == "4x4":
                    score_change += 4
        
        if self.our_position == "OUR_CLOSET":
            if self.our_held_block == "2x2":
                score_change -= 16*0.75
            elif self.our_held_block == "2x4":
                score_change -= 8*0.75
            elif self.our_held_block == "4x4":
                score_change -= 4*0.75
        elif self.our_position == "OUR_ROOM":
            if self.our_held_block == "2x2":
                score_change -= 4*0.75
            elif self.our_held_block == "2x4":
                score_change -= 2*0.75
            elif self.our_held_block == "4x4":
                score_change -= 1*0.75
        elif self.our_position == "HALLWAY":
            if self.our_held_block == "2x2":
                score_change -= 16
            elif self.our_held_block == "2x4":
                score_change -= 8
            elif self.our_held_block == "4x4":
                score_change -= 4
        elif self.our_position == "THEIR_ROOM":
            if self.our_held_block == "2x2":
                score_change += 4*0.75
            elif self.our_held_block == "2x4":
                score_change += 2*0.75
            elif self.our_held_block == "4x4":
                score_change += 1*0.75
        elif self.our_position == "THEIR_CLOSET":
            if self.our_held_block == "2x2":
                score_change += 16*0.75
            elif self.our_held_block == "2x4":
                score_change += 8*0.75
            elif self.our_held_block == "4x4":
                score_change += 4*0.75

        if self.their_position == "OUR_CLOSET":
            if self.their_held_block == "2x2":
                score_change -= 16*0.75
            elif self.their_held_block == "2x4":
                score_change -= 8*0.75
            elif self.their_held_block == "4x4":
                score_change -= 4*0.75
        elif self.their_position == "OUR_ROOM":
            if self.their_held_block == "2x2":
                score_change -= 4*0.75
            elif self.their_held_block == "2x4":
                score_change -= 2*0.75
            elif self.their_held_block == "4x4":
                score_change -= 1*0.75
        elif self.their_position == "HALLWAY":
            if self.their_held_block == "2x2":
                score_change += 16*0.75
            elif self.their_held_block == "2x4":
                score_change += 8*0.75
            elif self.their_held_block == "4x4":
                score_change += 4*0.75
        elif self.their_position == "THEIR_ROOM":
            if self.their_held_block == "2x2":
                score_change += 4*0.75
            elif self.their_held_block == "2x4":
                score_change += 2*0.75
            elif self.their_held_block == "4x4":
                score_change += 1*0.75
        elif self.their_position == "THEIR_CLOSET":
            if self.their_held_block == "2x2":
                score_change += 16*0.75
            elif self.their_held_block == "2x4":
                score_change += 8*0.75
            elif self.their_held_block == "4x4":
                score_change += 4*0.75
        
        self.current_score = score_change
        return self.current_score
    
    def get_available_actions(self, us=True):
        
        action_list = []

        if us:
            curr_pos = self.our_position
            curr_held = self.our_held_block
        else:
            curr_pos = self.their_position
            curr_held = self.their_held_block

        if curr_pos == "OUR_CLOSET":
            action_list.append(Action.MOVE_OUR_ROOM)
            if curr_held is None:
                if any(bp[1] == "OUR_CLOSET" and bp[0] == "2x2" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x2)
                if any(bp[1] == "OUR_CLOSET" and bp[0] == "2x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x4)
                if any(bp[1] == "OUR_CLOSET" and bp[0] == "4x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_4x4)
            else:
                action_list.append(Action.DROP_BLOCK)
        elif curr_pos == "OUR_ROOM":
            action_list.append(Action.MOVE_OUR_CLOSET)
            action_list.append(Action.MOVE_HALLWAY)
            if curr_held is None:
                if any(bp[1] == "OUR_ROOM" and bp[0] == "2x2" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x2)
                if any(bp[1] == "OUR_ROOM" and bp[0] == "2x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x4)
                if any(bp[1] == "OUR_ROOM" and bp[0] == "4x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_4x4)
            else:
                action_list.append(Action.DROP_BLOCK)
        elif curr_pos == "HALLWAY":
            action_list.append(Action.MOVE_OUR_ROOM)
            action_list.append(Action.MOVE_THEIR_ROOM)
        elif curr_pos == "THEIR_ROOM":
            action_list.append(Action.MOVE_HALLWAY)
            action_list.append(Action.MOVE_THEIR_CLOSET)
            if curr_held is None:
                if any(bp[1] == "THEIR_ROOM" and bp[0] == "2x2" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x2)
                if any(bp[1] == "THEIR_ROOM" and bp[0] == "2x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x4)
                if any(bp[1] == "THEIR_ROOM" and bp[0] == "4x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_4x4)
            else:
                action_list.append(Action.DROP_BLOCK)
        elif curr_pos == "THEIR_CLOSET":
            action_list.append(Action.MOVE_THEIR_ROOM)
            if curr_held is None:
                if any(bp[1] == "THEIR_CLOSET" and bp[0] == "2x2" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x2)
                if any(bp[1] == "THEIR_CLOSET" and bp[0] == "2x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_2x4)
                if any(bp[1] == "THEIR_CLOSET" and bp[0] == "4x4" for bp in self.block_positions):
                    action_list.append(Action.PICKUP_BLOCK_4x4)
            else:
                action_list.append(Action.DROP_BLOCK)
        
        return action_list
    
    def get_next_state(self, action, us=True):

        next_state = self.copy()

        if us:
            if action == Action.PICKUP_BLOCK_2x2:
                next_state.block_positions.remove(("2x2", self.our_position))
                next_state.our_held_block = "2x2"
            elif action == Action.PICKUP_BLOCK_2x4:
                next_state.block_positions.remove(("2x4", self.our_position))
                next_state.our_held_block = "2x4"
            elif action == Action.PICKUP_BLOCK_4x4:
                next_state.block_positions.remove(("4x4", self.our_position))
                next_state.our_held_block = "4x4"
            elif action == Action.DROP_BLOCK:
                next_state.our_held_block = None
                next_state.block_positions.append((self.our_held_block, self.our_position))
            elif action == Action.MOVE_OUR_CLOSET:
                next_state.our_position = "OUR_CLOSET"
            elif action == Action.MOVE_OUR_ROOM:
                next_state.our_position = "OUR_ROOM"
            elif action == Action.MOVE_HALLWAY:
                next_state.our_position = "HALLWAY"
            elif action == Action.MOVE_THEIR_ROOM:
                next_state.our_position = "THEIR_ROOM"
            elif action == Action.MOVE_THEIR_CLOSET:
                next_state.our_position = "THEIR_CLOSET"

        else:
            if action == Action.PICKUP_BLOCK_2x2:
                next_state.block_positions.remove(("2x2", self.their_position))
                next_state.their_held_block = "2x2"
            elif action == Action.PICKUP_BLOCK_2x4:
                next_state.block_positions.remove(("2x4", self.their_position))
                next_state.their_held_block = "2x4"
            elif action == Action.PICKUP_BLOCK_4x4:
                next_state.block_positions.remove(("4x4", self.their_position))
                next_state.their_held_block = "4x4"
            elif action == Action.DROP_BLOCK:
                next_state.their_held_block = None
                next_state.block_positions.append((self.our_held_block, self.their_position))
            elif action == Action.MOVE_OUR_CLOSET:
                next_state.their_position = "OUR_CLOSET"
            elif action == Action.MOVE_OUR_ROOM:
                next_state.their_position = "OUR_ROOM"
            elif action == Action.MOVE_HALLWAY:
                next_state.their_position = "HALLWAY"
            elif action == Action.MOVE_THEIR_ROOM:
                next_state.their_position = "THEIR_ROOM"
            elif action == Action.MOVE_THEIR_CLOSET:
                next_state.their_position = "THEIR_CLOSET"

        next_state.update_and_get_score()
        return next_state

