from tqdm import tqdm

from GridWorld import GridWorld
import numpy as np
from Logging import *

class RLAgent(object):
    """
        Reinforcement Learning Agent Model for training/testing
        with Tabular function approximation

    """

    def __init__(self, env):
        self.env = env
        self.size = env.get_size()
        self.n_a = len(env.get_actions())
        # self.Q table including the surrounding border
        self.Q = np.zeros((self.size[0], self.size[1], self.n_a))

    def epsilon_greed(self, epsilon, s):
        if np.random.rand() < epsilon:
            return np.random.randint(len(self.env.get_actions()))
        else:
            return self.greedy(s)

    def greedy(self, s):
        return np.argmax(self.Q[tuple(s)])

    def coord_convert(self, s, sz):
        return [s[1], sz[0] - s[0] - 1]

    def train_sarsa(self, start, **params):

        # parameters
        gamma = params.pop('gamma', 0.99)
        alpha = params.pop('alpha', 0.1)
        epsilon = params.pop('epsilon', 0.1)
        maxiter = params.pop('maxiter', 1000)
        maxstep = params.pop('maxstep', 1000)

        # init self.Q matrix
        self.Q[...] = 0
        self.Q = self.env.exclude_invalid_regions(self.Q)

        # online train
        # rewards and step trace
        rtrace = []
        steps = []
        trace = 0
        for j in tqdm(range(maxiter)):
            self.env.init(start)
            # Logging.show_q(self.Q)
            s = self.env.get_cur_state()
            # selection an action
            a = self.epsilon_greed(epsilon, s)

            rewards = []
            # This changes the position of the robot location to matplot coord
            trace = np.array(self.coord_convert(s, self.size))
            # run simulation for max number of steps
            step = 0
            for step in range(maxstep):
                # move
                r = self.env.next(a)
                s1 = self.env.get_cur_state()
                a1 = self.epsilon_greed(epsilon, s1)

                rewards.append(r)
                trace = np.vstack((trace, self.coord_convert(s1, self.size)))

                # This is SARSA control
                v = self.Q[tuple(list(s) + list([a]))] + alpha * (r + gamma * self.Q[tuple(list(s1) + list([a1]))] -
                                                                  self.Q[tuple(list(s) + list([a]))])
                self.Q[tuple(list(s) + list([a]))] = v

                if self.env.is_goal():  # reached the goal
                    # Setting the Goal location to 0, allows for training
                    # the best path.
                    # self.normalize()
                    self.Q[tuple(list(s1) + list([a1]))] = 0
                    break

                s = s1
                a = a1

            rtrace.append(np.sum(rewards))
            steps.append(step + 1)

        # Logging.show_q(self.Q)
        return rtrace, steps, trace  # last trace of trajectory

    def train_q(self, start, **params):

        # parameters
        gamma = params.pop('gamma', 0.99)
        alpha = params.pop('alpha', 0.1)
        epsilon = params.pop('epsilon', 0.1)
        maxiter = params.pop('maxiter', 1000)
        maxstep = params.pop('maxstep', 1000)

        # init self.Q matrix
        self.Q[...] = 0
        self.Q = self.env.exclude_invalid_regions(self.Q)

        # online train
        # rewards and step trace
        rtrace = []
        steps = []
        trace = 0
        for j in tqdm(range(maxiter)):
            self.env.init(start)
            # Logging.show_q(self.Q)
            s = self.env.get_cur_state()
            # selection an action
            a = self.epsilon_greed(epsilon, s)

            rewards = []
            # This changes the position of the robot location to matplot coord
            trace = np.array(self.coord_convert(s, self.size))
            # run simulation for max number of steps
            step = 0
            for step in range(maxstep):
                # move
                r = self.env.next(a)
                s1 = self.env.get_cur_state()
                a = self.epsilon_greed(epsilon, s1)

                rewards.append(r)
                trace = np.vstack((trace, self.coord_convert(s1, self.size)))

                # This is SARSA control
                v = self.Q[s[0], s[1], a] + alpha * (r + gamma * np.max(self.Q[s1[0], s1[1]]) - self.Q[s[0], s[1], a])
                self.Q[s[0], s[1], a] = v

                if self.env.is_goal():  # reached the goal
                    # Setting the Goal location to 0, allows for training
                    # the best path.
                    # self.normalize()
                    self.Q[s1[0], s1[1], a] = 0
                    break

                s = s1

            rtrace.append(np.sum(rewards))
            steps.append(step + 1)

        # Logging.show_q(self.Q)
        return rtrace, steps, trace  # last trace of trajectory

    def test(self, start, maxstep=1):
        # Init trace array for tracking the agent movement
        trace = np.array(self.coord_convert(start, self.size))

        # Initialize the starting environment
        self.env.init(start)
        # Show the starting state
        s = self.env.get_cur_state()
        # selection an action
        a = np.argmax(self.Q[s[0], s[1]])
        temp = np.max(self.Q[s[0], s[1]])

        for i in tqdm(range(maxstep)):
            # Move to to the next best state based on the current action
            r = self.env.next(a)
            s = self.env.get_cur_state()
            a = np.argmax(self.Q[s[0], s[1]])
            temp = np.max(self.Q[s[0], s[1]])
            # Log the agent movement
            trace = np.vstack((trace, self.coord_convert(s, self.size)))

            if self.env.is_goal():  # reached the goal
                # Setting the Goal location to 0, allows for training
                # the best path.
                # self.normalize()
                print('Goal is found!')
                break

        return trace

if __name__ == '__main__':
    """
    Note: 
    import matplotlib.pyplot as plt
    plt.imshow(np.max(self.Q, axis=2))
    plt.show()
    """
    env = GridWorld('grid.txt')

    env.print_map()

    model = RLAgent(env)
    rtrace, steps, trace = model.train_sarsa(start=None, gamma=.99, alpha=.01, epsilon=0.1,
                                             maxiter=1000, maxstep=1000)
    trace = model.test([0,0])

    Logging.plot_train(model, rtrace, steps, trace, [0,0], env)



    print([2, 3], env.check_state([2, 3]))
    print([0, 0], env.check_state([0, 0]))
    print([3, 4], env.check_state([3, 4]))
    print([10, 3], env.check_state([10, 3]))

    env.init([0, 0])
    print(env.next(1))  # right
    print(env.next(3))  # down
    print(env.next(0))  # left
    print(env.next(2))  # up
    print(env.next(2))  # up
