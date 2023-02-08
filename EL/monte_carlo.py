import collections
import gym
from frozen_lake_util import show_q_value
from el_agent import ElAgent


class MonteCarloAgent(ElAgent):
    def __init__(self, epsilon=0.1):
        super().__init__(epsilon=epsilon)

    def learn(
        self, env, episode_count=1000, gamma=0.99, render=False, report_interval=50
    ):
        self.init_log()

        actions = list(range(env.action_space.n))
        self.Q = collections.defaultdict(lambda: [0] * len(actions))
        N = collections.defaultdict(lambda: [0] * len(actions))

        for episode in range(episode_count):
            s = env.reset()
            done = False
            experience = []
            while not done:
                if render:
                    env.render()

                a = self.policy(s, actions=actions)
                n_state, reward, done, info = env.step(a)
                experience.append({"state": s, "action": a, "reward": reward})
                s = n_state

            else:
                self.log(reward)

            for index1, result in enumerate(experience):
                state = result["state"]
                action = result["action"]

                G = 0
                gamma_coef = 1
                for index2 in range(index1, len(experience)):
                    reward_info = experience[index2]
                    reward = reward_info["reward"]
                    G += gamma_coef * reward

                    gamma_coef *= gamma

                N[state][action] += 1
                alpha = 1 / N[state][action]
                self.Q[state][action] += alpha * (G - self.Q[state][action])

            if episode != 0 and episode % report_interval == 0:
                self.show_reward_log(episode=episode)


def train():
    agent = MonteCarloAgent(epsilon=0.1)
    env = gym.make("FrozenLakeEasy-v0")

    agent.learn(env, episode_count=500)
    show_q_value(agent.Q)
    agent.show_reward_log()


if __name__ == "__main__":
    train()
