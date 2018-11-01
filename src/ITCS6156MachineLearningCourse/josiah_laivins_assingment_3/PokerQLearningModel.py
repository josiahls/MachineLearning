from tqdm import tqdm
import numpy as np

from BlackJack import PokerEnvWrapper
from Logging import *

class RLAgent:
    """
        Reinforcement Learning Agent Model for training/testing
        with Tabular function approximation

    """

    def __init__(self, env: PokerEnvWrapper):
        self.env = env
        self.size = env.get_size()
        self.n_a = len(env.get_actions())
        # self.Q table including the surrounding border
        q_shape = tuple(list(self.size) + list([self.n_a]))
        self.Q = np.zeros(q_shape)

    def epsilon_greed(self, epsilon, s):
        if np.random.rand() < epsilon:
            return np.random.randint(len(self.env.get_actions()))
        else:
            return self.greedy(s)

    def greedy(self, s):
        return np.argmax(self.Q[tuple(s)])

    # def coord_convert(self, s, sz):
    #     return [s[1], sz[0] - s[0] - 1]

    def train_sarsa(self, start, **params):

        # parameters
        gamma = params.pop('gamma', 0.99)
        alpha = params.pop('alpha', 0.1)
        epsilon = params.pop('epsilon', 0.1)
        maxiter = params.pop('maxiter', 1000)

        # init self.Q matrix
        self.Q[...] = 0
        self.Q = self.env.exclude_invalid_regions(self.Q)

        # online train
        # rewards and step trace
        rtrace = []
        steps = []
        for j in tqdm(range(maxiter)):
            self.env.init(start)
            # Logging.show_q(self.Q)
            s = self.env.get_cur_state()
            # selection an action
            a = self.epsilon_greed(epsilon, s)

            rewards = []
            step = 0
            # run simulation for max number of steps
            while not self.env.poker_env.deal():
                step += 1
                # move
                r = self.env.next(a)
                s1 = self.env.get_cur_state()
                a1 = self.epsilon_greed(epsilon, s1)

                rewards.append(r)

                # This is SARSA control
                v = self.Q[tuple(list(s)+list([a]))] + alpha * \
                    (r + gamma * self.Q[tuple(list(s1)+list([a1]))] - self.Q[tuple(list(s)+list([a]))])

                self.Q[tuple(list(s)+list([a]))] = v

                s = s1
                a = a1

            rtrace.append(np.sum(rewards))
            steps.append(step + 1)

        return rtrace, steps  # last trace of trajectory

    def train_q(self, start, **params):

        # parameters
        gamma = params.pop('gamma', 0.99)
        alpha = params.pop('alpha', 0.1)
        epsilon = params.pop('epsilon', 0.1)
        maxiter = params.pop('maxiter', 1000)

        # init self.Q matrix
        self.Q[...] = 0
        self.Q = self.env.exclude_invalid_regions(self.Q)

        # online train
        # rewards and step trace
        rtrace = []
        steps = []
        for j in tqdm(range(maxiter)):
            self.env.init(start)
            # Logging.show_q(self.Q)
            s = self.env.get_cur_state()
            # selection an action
            a = self.epsilon_greed(epsilon, s)

            rewards = []
            step = 0
            # run simulation for max number of steps
            while not self.env.poker_env.deal():
                step += 1
                # move
                r = self.env.next(a)
                s1 = self.env.get_cur_state()
                a = self.epsilon_greed(epsilon, s1)

                rewards.append(r)

                # This is SARSA control
                v = self.Q[tuple(list(s) + list([a]))] + alpha * \
                    (r + gamma * np.max(self.Q[tuple(list(s1))]) - self.Q[tuple(list(s) + list([a]))])

                self.Q[tuple(list(s) + list([a]))] = v

                s = s1

            rtrace.append(np.sum(rewards))
            steps.append(step + 1)

        return rtrace, steps  # last trace of trajectory


    def test(self, maxstep=1):
        wins = []
        loses = []

        for i in tqdm(range(maxstep)):
            # Play one game

            # Initialize the starting environment
            self.env.init(None)
            # Show the starting state
            s = self.env.get_cur_state()
            # selection an action
            a = np.argmax(self.Q[tuple(s)])
            temp = np.max(self.Q[tuple(s)])

            while not self.env.poker_env.deal():
                # Move to to the next best state based on the current action
                r = self.env.next(a)
                s = self.env.get_cur_state()
                a = np.argmax(self.Q[tuple(s)])
                temp = np.max(self.Q[tuple(s)])

            winner = self.env.poker_env.return_winner(self.env.poker_env.all_players)

            if self.env.player_name in winner:
                wins.append(1)
                loses.append(0)
            else:
                loses.append(1)
                wins.append(0)

        return wins, loses
