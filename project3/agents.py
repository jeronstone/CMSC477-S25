from util import *
from state import *

# choose the immediate action with the highest score
class ReflexAgent():
    def __init__(self, curr_state: State):
        self.curr_state: State = curr_state

    def choose_action(self):
        actions = self.curr_state.get_available_actions()

        best_action = None
        best_score = float('-inf')
        for action in actions:
            next_state = self.curr_state.get_next_state(action, us=True)
            if next_state.current_score > best_score:
                best_action = action
                best_score = next_state.current_score
        
        return best_action
    
    def perform_action(self, action):
        self.curr_state = self.curr_state.get_next_state(action, us=True)

# choose the best action assuming the other robot is perfect
class MiniMaxAgent():
    def __init__(self, curr_state: State, search_depth: int):
        self.curr_state: State = curr_state
        self.search_depth: int = search_depth

    def min_layer(self, curr_state: State, depth: int):
        if depth + 1 >= self.search_depth:
            return curr_state.current_score, None
        min_score = float('inf')
        min_action = None
        actions = curr_state.get_available_actions(us=False)
        for action in actions:
            next_state = curr_state.get_next_state(action, us=False)
            next_score, _ = self.max_layer(next_state, depth + 1)
            if next_score < min_score:
                min_score = next_score
                min_action = action
        return min_score, min_action
    
    def max_layer(self, curr_state: State, depth: int):
        if depth + 1 >= self.search_depth:
            return curr_state.current_score, None
        max_score = float('-inf')
        max_action = None
        actions = curr_state.get_available_actions(us=True)
        for action in actions:
            next_state = curr_state.get_next_state(action, us=True)
            next_score, _ = self.min_layer(next_state, depth + 1)
            if next_score > max_score:
                max_score = next_score
                max_action = action
        return max_score, max_action
        
    def choose_action(self):
        best_score, best_action = self.max_layer(self.curr_state, 0)
        print(f"found best action {best_action} with best score {best_score}")
        return best_action
    
    def perform_action(self, action):
        self.curr_state = self.curr_state.get_next_state(action)

# choose the best action assuming the other robot is not perfect
class ExpectiMaxAgent():
    def __init__(self, curr_state: State, search_depth: int):
        self.curr_state: State = curr_state
        self.search_depth: int = search_depth

    def expecti_layer(self, curr_state: State, depth: int):
        if depth + 1 >= self.search_depth:
            return curr_state.current_score, None
        avg_score = 0.0
        avg_action = None
        actions = curr_state.get_available_actions(us=False)
        if len(actions) < 1:
            return 0.0, None
        for action in actions:
            next_state = curr_state.get_next_state(action, us=False)
            next_score, _ = self.max_layer(next_state, depth + 1)
            avg_score += next_score
        return avg_score/len(actions), None
    
    def max_layer(self, curr_state: State, depth: int):
        if depth + 1 >= self.search_depth:
            return curr_state.current_score, None
        max_score = float('-inf')
        max_action = None
        actions = curr_state.get_available_actions(us=True)
        for action in actions:
            next_state = curr_state.get_next_state(action, us=True)
            next_score, _ = self.expecti_layer(next_state, depth + 1)
            if next_score > max_score:
                max_score = next_score
                max_action = action
        return max_score, max_action
        
    def choose_action(self):
        best_score, best_action = self.max_layer(self.curr_state, 0)
        print(f"found best action {best_action} with best score {best_score}")
        return best_action
    
    def perform_action(self, action):
        self.curr_state = self.curr_state.get_next_state(action)
