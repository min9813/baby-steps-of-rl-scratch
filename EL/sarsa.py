import collections
import el_agent
import gym
import frozen_lake_util


class SARSAAgent(el_agent.ElAgent):
    def __init__(self, epsilon):
        super().__init__(epsilon)

    def learn(
        self,
        env,
        render=False,
        episode_count=1000,
        gamma=0.9,
        learning_rate=0.1,
        report_interval=50,
    ):

        actions = list(range(env.action_space.n))
        self.Q = collections.defaultdict(lambda: [0] * len(actions))

        for e in range(episode_count):
            s = env.reset()

            done = False
            a = self.policy(s, actions=actions)

            while not done:
                if render:
                    env.render()

                n_state, reward, done, info = env.step(a)
                n_action = self.policy(n_state, actions=actions)

                # gain = reward + gamma * max(self.Q[n_state])
                gain = reward + gamma * self.Q[n_state][n_action]
                estimated = self.Q[s][a]

                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state
                a = n_action

            else:
                self.log(reward=reward)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train():
    epsilon = 0.1
    agent = SARSAAgent(epsilon=epsilon)
    env = gym.make("FrozenLakeEasy-v0")

    episode_count = 10000
    learning_rate = 0.01
    agent.learn(env, episode_count=episode_count, learning_rate=learning_rate)
    frozen_lake_util.show_q_value(Q=agent.Q)
    agent.show_reward_log()


if __name__ == "__main__":
    train()
