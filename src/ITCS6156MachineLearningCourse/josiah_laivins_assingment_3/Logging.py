import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Note this is the needed because pycharm doesnt seem to handle live graphs in the sci view very well
import matplotlib.pyplot as plt

class Logging:
    @staticmethod
    def show_q_interactive(Q: np.array):
        """
        Displays Q as a N dimensional image, most likely 2D

        :param Q:
        :return:
        """
        plt.ion()
        Q = np.copy(Q)
        Q[Q == -np.inf] = -10
        plt.imshow(np.max(Q, axis=2))
        plt.draw()
        plt.pause(0.01)
        print(f'{np.max(Q, axis=2)}')

    @staticmethod
    def show_q(Q: np.array):
        """
        Displays Q as a N dimensional image, most likely 2D

        :param Q:
        :return:
        """
        Q = np.copy(Q)
        Q[Q == -np.inf] = -10
        plt.imshow(np.max(Q, axis=2))
        plt.show(blocking=False)

    @staticmethod
    def plot_trace(agent, start, trace, env, title="test trajectory"):
        plt.plot(trace[:, 0], trace[:, 1], "ko-")
        plt.text(env.goal_pos[1], agent.size[0] - env.goal_pos[0] - 1, 'G')
        plt.text(start[1], agent.size[0] - start[0] - 1, 'S')
        plt.xlim([0, agent.size[1]])
        plt.ylim([0, agent.size[0]])
        # plt.show(blocking=True)