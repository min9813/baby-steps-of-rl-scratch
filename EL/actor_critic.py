import numpy as np
import gym
from el_agent import ELAgent
from frozen_lake_util import show_q_value


class Actor(ELAgent):
    def __init__(self, env) -> None:
        super().__init__(epsilon=1)

        self.env = env
        nrow = env.observation_space.n
        ncol = env.action_space.n

        self.actions = list(range(env.action_space.n))
        self.Q = np.random.uniform(0, 1, nrow * ncol).reshape((nrow, ncol))

    def softmax(self, x):
        x_exp = np.exp(x)
        return x_exp / np.sum(x_exp, axis=0)

    def policy(self, s):
        a = np.random.choice(self.actions, 1, p=self.softmax(self.Q[s]))
        return a[0]


class Critic:
    def __init__(self, env) -> None:
        states = env.observation_space.n
        self.V = np.zeros(states)


class ActorCritic:
    def __init__(self, actor_class, critic_class) -> None:
        self.actor_clsss = actor_class
        self.critic_class = critic_class

    def train(
        self,
        env,
        episode_count=1000,
        gamma=0.9,
        learning_rate=0.1,
        render=False,
        report_interval=50,
    ):
        actor = self.actor_clsss(env)
        critic = self.critic_class(env)

        actor.init_log()
        for e in range(episode_count):
            s = env.reset()
            done = False

            while not done:
                if render:
                    env.render()

                a = actor.policy(s)
                n_state, reward, done, info = env.step(a)

                gain = reward + gamma * critic.V[n_state]
                estimate = critic.V[s]
                td = gain - estimate

                actor.Q[s][a] += learning_rate * td
                critic.V[s] += learning_rate * td

                s = n_state

            else:
                actor.log(reward)

            if e != 0 and e % report_interval != 0:
                actor.show_reward_log(episode=e)

        return actor, critic


def train():
    trainer = ActorCritic(Actor, Critic)
    env = gym.make("FrozenLakeEasy-v0")

    episode_num = 3000
    actor, critic = trainer.train(env=env, episode_count=episode_num)

    print('ok')
    show_q_value(actor.Q)
    print('ok')
    actor.show_reward_log()


if __name__ == "__main__":
    train()
