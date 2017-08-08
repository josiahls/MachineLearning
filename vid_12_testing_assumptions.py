"""
Here we will create a regression algorithm in
pure python and implimenting r squared
"""
import random
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

# this is if you want things easy
# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


# NEW CODE FOR TESTING
# this creates a nice array based on several params
def create_dataset(howMany, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(howMany):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    # print(m)
    # below is the easier way of calculating the
    # slope
    m = ((mean(xs) * mean(ys) - mean(xs * ys)) /
         (mean(xs) * mean(xs) - mean(xs * xs)))
    # print(m)
    # added text to push

    b = mean(ys) - m * mean(xs)
    return m, b


def squared_error(ys_original, ys_line):
    return sum((ys_line - ys_original) ** 2)


def coefficient_of_determination(ys_original, ys_line):
    y_mean_line = [mean(ys_original) for y in ys_original]
    squared_error_regr = squared_error(ys_original, ys_line)
    squared_error_y_mean = squared_error(ys_original, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


xs, ys = create_dataset(100, 90, 2, correlation=False)

m, b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m * x) + b for x in xs]
print(m, b)

predict_x = 8
predict_y = (m * predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y)
plt.plot(xs, regression_line)
plt.show()
