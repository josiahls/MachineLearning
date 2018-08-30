"""
so remember that vecX * vecW + b is our hyperplane and there are 2 versions:
one has the equation set to 1 for a positive class and one has -1 for a
negative class.

So if we had vecX * vecW + b = 0.98 then we would have a vecX that is
positive.

So we still need to find min magW and max b
So we create:

class (known features * W + b) >= 1 with
lowest W and largest b

So a convex problem is a problem that is a line
that might be like a bowl where was are trying to find
the lowest of the bowl. This line is vectors of W, and we just
take vectors Yi (Xi * W + b) >= 1 and try each feature and find
lowest value

BTW the norm is vecW([5,3]) * sqrt(5^2 + 3^2)

So lets say we have
vecW = [5,5] and we want to find any b
that satisfies Yi (vecXi * vecW + b) >= 1
We can do this by starting with the largest b, and start
steping down till the equation is satisfied and find the b
that solves all variations of vecW [5,5], [-5,5], [-5,-5], [5,-5]

This can be optimized by increasing the step size of b, and modifiy
this step size till you find the global min of the equation.
We can just save each value into a dictionary mag = {magW:[vecW b]}
and go back through and find the smallest value

Basically if we find a data point that gets Yi(Xi * vecW + b) close to 1
then we have a good hyperplane.

optimization is a major field. This specific problem is a convex optimization
problem which I think is a basically a bottom of the hill probem or
lowest, highest point problem.
"""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


class SupportVectorMachine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        self.data = 0
        self.max_feature_value = 0
        self.min_feature_value = 0

        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # train
    # noinspection PyAttributeOutsideInit
    def fit(self, data):
        self.data = data
        # {||w||: [w,b]}
        opt_dict = {}

        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        all_data = []
        # yi is what ever the pre determined class is -1 or 1
        for yi in self.data:
            # feature set is the set of features
            for featureset in self.data[yi]:
                # each feature is added to a long list
                for feature in featureset:
                    all_data.append(feature)

        # we get the extremes of all of the features
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        # noinspection PyUnusedLocal
        all_data = None

        # support vectors yi(xi.w + b)

        # step sizes will be used to find the max b allowed
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense
                      self.max_feature_value * 0.001
                      ]

        # extremely expensive
        b_range_multiple = 5

        # we dont need to take small steps
        # with b as we do w
        b_multiple = 5

        latest_optimum = self.max_feature_value * 10
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:

                # we use arange to specify step
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # weakest link in the SVM fundumentally
                        # smo attemps to fix this a bit
                        # yi (xi.w + b) >= 1
                        #
                        # add a break here later..
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                # if even one of the variations is
                                # invalid, then we throw out that vector
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    # print(i, xi, ':',yi * (np.dot(w_t, xi) + b))

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print('optimized a step.')
                else:
                    # w = [5,5]
                    # step = 1
                    # w - step = [4,4]
                    w = w - step

            # these are the norms or the magnitudes
            norms = sorted([n for n in opt_dict])

            # the most optimum is the smallest norm
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]

            latest_optimum = opt_choice[0][0] + step * 2

        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi, ':',yi * (np.dot(self.w, xi) + self.b))

    # noinspection PyArgumentList
    def predict(self, features):
        # sign of x.w+b
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])

        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w + b
        # psv =
        # nsv = -1
        # dec = 0
        # note that this function is just for visuals trololol
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # w.x +b = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # w.x +b = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # w.x +b = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8]]),
             1: np.array([[5, 1], [6, -1], [7, 3]])}

svm = SupportVectorMachine()
svm.fit(data_dict)

predict_us = [[0, 10], [1, 3], [3, 4], [3, 5], [5, 5], [5, 6], [6, -5], [5, 8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()
