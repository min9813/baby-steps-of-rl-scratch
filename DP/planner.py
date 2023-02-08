class Planner:
    def __init__(self, env):
        self.env = env
        self.log = []

    def initialize(self):
        self.env.reset()
        self.log = []

    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception("Planner have to implements plan method.")

    def transitions_at(self, state, action):
        transition_probs = self.env.transit_func(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            reward, _ = self.env.reward_func(next_state)
            yield prob, reward, next_state

    def dict_to_grid(self, state_reward_dict):
        grid = []
        for i in range(self.env.row_num):
            row = [0] * self.env.column_num
            grid.append(row)
        for s in state_reward_dict:
            grid[s.row][s.column] = state_reward_dict[s]

        return grid


class ValueIterationPlanner(Planner):
    def __init__(self, env):
        super().__init__(env)

    def plan(self, gamma=0.99, threshould=0.0001):
        self.initialize()
        V = {}

        for s in self.env.states:
            V[s] = 0

        while True:
            delta = 0
            self.log.append(self.dict_to_grid(V))

            for now_state in V:
                if not self.env.can_action_at(s):
                    continue

                max_v = 0
                for a in self.env.actions:
                    # next_state, reward, done = self.env.transit(state=v, action=a)
                    now_value = 0
                    for prob, reward, next_state in self.transitions_at(
                        state=now_state, action=a
                    ):
                        now_value += prob * (reward + gamma * V[next_state])

                    max_v = max(max_v, now_value)

                # print(
                #     "now state={}, max_v={}, V[s]={}".format(
                #         now_state, max_v, V[now_state]
                #     )
                # )
                delta = max(delta, abs(max_v - V[now_state]))
                V[now_state] = max_v

            # print("delta={}".format(delta))

            if delta < threshould:
                break
        V_grid = self.dict_to_grid(V)

        return V_grid


class PolicyIterationPlanner(Planner):
    def __init__(self, env):
        super().__init__(env)

    def initialize(self):
        super().initialize()

        states = self.env.states
        actions = self.env.actions

        self.policy = {}
        for state in states:
            self.policy[state] = {}
            for action in actions:
                self.policy[state][action] = 1 / len(actions)

    def estimate_by_policy(self, gamma, threshould):
        V = {}
        for state in self.env.states:
            V[state] = 0

        while True:
            delta = 0
            for now_state in V:
                # for a in self.env.actions:
                policy = self.policy[now_state]

                total_state_value = 0
                for action in policy:
                    action_prob = policy[action]
                    this_s_a_value = 0
                    for prob, reward, next_state in self.transitions_at(
                        state=now_state, action=action
                    ):
                        this_s_a_value += (
                            action_prob * prob * (reward + gamma * V[next_state])
                        )

                    total_state_value += this_s_a_value

                delta = max(abs(V[now_state] - total_state_value), delta)
                V[now_state] = total_state_value
            if delta < threshould:
                break

        return V

    def plan(self, gamma=0.9, threshould=0.0001):
        self.initialize()
        states = self.env.states
        actions = self.env.actions

        while True:
            V = self.estimate_by_policy(gamma=gamma, threshould=threshould)
            self.log.append(self.dict_to_grid(V))

            is_update = False
            for now_state in states:
                expected_values = {}
                policy = self.policy[now_state]
                policy_action = take_max_action(action2value=policy)

                for action in actions:
                    action_value = 0
                    for prob, reward, next_state in self.transitions_at(
                        state=now_state, action=action
                    ):
                        action_value += prob * (reward + V[next_state])

                    expected_values[action] = action_value

                best_action = take_max_action(expected_values)

                if policy_action != best_action:
                    is_update = True

                for a in policy:
                    if a == best_action:
                        prob = 1
                    else:
                        prob = 0

                    policy[a] = prob

            if not is_update:
                break

        V_grid = self.dict_to_grid(V)
        return V_grid


def take_max_action(action2value):
    best_action = max(action2value, key=lambda x: action2value[x])
    return best_action
