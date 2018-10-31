import numpy as np
import matplotlib
# matplotlib.use("TkAgg")  # Note this is the needed because pycharm doesnt seem to handle live graphs in the sci view very well
import matplotlib.pyplot as plt

def matprint(mat:np.array, fmt="g"):
    np.set_printoptions(precision=3)
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


class Logging:
    @staticmethod
    def show_q_interactive(Q: np.array, show_max=True):
        """
        Displays Q as a N dimensional image, most likely 2D

        :param Q:
        :return:
        """
        plt.ion()
        Q = np.copy(Q)
        Q[Q == -np.inf] = 0
        if show_max:
            plt.imshow(np.max(Q, axis=2))
        else:
            plt.imshow(np.min(Q, axis=2))
        plt.draw()
        plt.pause(0.01)
        # print('\nMax Q:\n')
        # matprint(np.max(Q, axis=2))
        # print('\nMin Q:\n')
        # matprint(np.min(Q, axis=2))

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
        plt.show()

    @staticmethod
    def plot_trace(agent, start, trace, env, title="test trajectory", axis=None):
        plt.text(env.goal_pos[1], agent.size[0] - env.goal_pos[0] - 1, 'G')
        plt.text(start[1], agent.size[0] - start[0] - 1, 'S')
        plt.xlim([0, agent.size[1]])
        plt.ylim([0, agent.size[0]])
        if axis is None:
            plt.plot(trace[:, 0], trace[:, 1], "ko-")
        else:
            axis.plot(trace[:, 0], trace[:, 1], "ko-")
        plt.plot()
        plt.ylabel(title)

    @staticmethod
    def plot_train(agent, rtrace, steps, trace: list=None, start: list=None, env=None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(221)
        ax.plot(rtrace)
        plt.plot()
        plt.ylabel("sum of rewards")
        ax1 = fig.add_subplot(222)
        ax1.plot(steps)
        plt.plot()
        plt.ylabel("# steps")

        if env is not None:
        # contour plot for agent.Q
            ax2 = fig.add_subplot(223)
            xs = range(agent.size[1])
            ys = range(agent.size[0])
            maxQ = np.max(agent.Q, axis=2)
            h_b = (maxQ == -np.inf)
            maxQ[h_b] = 0
            maxQ[h_b] = np.min(maxQ) - 100
            cs = plt.contourf(xs, ys[::-1], maxQ)
            plt.colorbar(cs)
            plt.text(env.goal_pos[1], agent.size[0] - env.goal_pos[0] - 1, 'G')
            plt.text(start[1], agent.size[0] - start[0] - 1, 'S')
            plt.plot()
            plt.ylabel("max agent.Q")

            # plot traces
            ax3 = fig.add_subplot(224)
            Logging.plot_trace(agent, start, trace, env, "trace of the last episode", axis=ax3)
            plt.ylabel("Agent Path Trace")
