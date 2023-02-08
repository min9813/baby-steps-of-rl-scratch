import numpy as np
import matplotlib.pyplot as plt


class ELAgent:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.Q = {}
        self.reward_log = []

    def policy(self, s, actions):
        p = np.random.random()

        if p < self.epsilon:
            return np.random.randint(len(actions))

        else:
            if s in self.Q and np.sum(self.Q[s]) != 0:
                return np.argmax(self.Q[s])

            else:
                return np.random.randint(len(actions))

    def init_log(self):
        self.reward_log = []

    def log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self, interval=50, episode=-1):
        if episode >= 0:
            rewards = self.reward_log[-interval:]

            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)

            print("At Episode {} average reward is {} +- {}".format(episode, mean, std))

        else:
            indices = range(0, len(self.reward_log), interval)

            means = []
            stds = []

            for i in indices:
                rewards = self.reward_log[i : i + interval]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))

            means = np.array(means)
            stds = np.array(stds)

            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds, alpha=0.1, color="g")
            plt.plot(
                indices,
                means,
                "o-",
                color="g",
                label="Rewards for each {} episode".format(interval),
            )
            plt.legend(loc="best")
            plt.show()
