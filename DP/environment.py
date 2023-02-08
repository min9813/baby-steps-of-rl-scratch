from enum import Enum
import numpy as np


class State:
    def __init__(self, row=0, column=0) -> None:
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    UP = (1, 0)
    DOWN = (-1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class Environment:
    def __init__(self, grid, move_prob=0.8) -> None:

        self.grid = grid
        self.agent_state = State()

        self.default_reward = -0.04

        self.move_prob = move_prob
        self.reset()

    @property
    def row_num(self):
        return len(self.grid)

    @property
    def column_num(self):
        return len(self.grid[0])

    @property
    def actions(self):
        actions = [
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        ]
        return actions

    @property
    def states(self):
        states = []
        for row in range(self.row_num):
            for col in range(self.column_num):
                if self.grid[row][col] != 9:
                    states.append(State(row, col))

        return states

    def reset(self):
        self.agent_state = State(self.row_num - 1, 0)
        pass

    def transit_func(self, state, action):
        transition_probs = {}

        if not self.can_action_at(state):
            return transition_probs

        other_action_prob = (1 - self.move_prob) * 0.5
        # print(action)

        # action_content = Action[str(action)]
        action_content = action.value
        opposite_action = Action((-action_content[0], -action_content[1]))

        next_state2prob = {}
        for next_action in self.actions:
            if next_action == opposite_action:
                prob = 0
            elif next_action == action:
                prob = self.move_prob
            else:
                prob = other_action_prob

            next_state = self._move(state=state, action=next_action)

            if next_action not in next_state2prob:
                next_state2prob[next_state] = prob
            else:
                next_state2prob[next_state] += prob

        return next_state2prob

    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0:
            return True

        else:
            return False

    def _move(self, state, action):
        if not self.can_action_at(state):
            raise ValueError("cannot move from {}".format(state))

        next_state = state.clone()
        next_state.row += action.value[0]
        next_state.column += action.value[1]

        if not (0 <= next_state.row < self.row_num):
            next_state = state

        if not (0 <= next_state.column < self.column_num):
            next_state = state

        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state):
        reward = self.default_reward

        if self.grid[state.row][state.column] == 1:
            reward = 1
            done = True

        elif self.grid[state.row][state.column] == -1:
            reward = -1
            done = True

        else:
            done = False

        return reward, done

    def transit(self, state, action):
        transition_prob = self.transit_func(state=state, action=action)

        state_list = []
        prob_list = []
        for state, prob in transition_prob.items():
            state_list.append(state)
            prob_list.append(prob)

        next_state = np.random.choice(a=state_list, p=prob_list, size=1)
        reward, done = self.reward_func(state=next_state)

        return next_state, reward, done
