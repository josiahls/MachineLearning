"""
x1 = np.array([5,3,2,2,5,2,1])
x2 = np.array([2,1,1,6,4,6,1])

print(np.dot(x1,x2)) dot and inner output the same value, but
                    inner is prefered
print(np.inner(x1,x2))

A kernal takes in 2 inputs and outputs a single result and is works
as a similarity function

Find the class of X is found by:
Y = sign(Wbar * Xbar+b)
the dot product of Wbar and Xbar will always output a 
scaler regardless of dimensions

You can swap Xbar with Zbar which is a vector with inf dimensions

W = sum(alpha* yi*xi)

L = sum(alphai) - (1/2)sum(alphai * alphaj * yi* yj * (xi * xj))

Example Kernal

K (x,xprime) = Z Dotproduct Zprime

Z is a function that is applied to its X counterpart:
Z = function(X)
Zprime = function(xprime)

Which can be:
Y Wbar(K(x,xprime))x+b as opposed to y=wbar*xbar+b

So feature set:
[x1,x2] convert to 2nd order polynomial:
X=[x1,x2]    Z = [1,x1,x2,x1^2,x2^2,x1*x2] which is now 2nd orderpoly
             Zprime is Z but with xprimes
So

K(x,xprime) = Z * Zprime = [1+x1*xprime1+x2*xprime2+x1^2*xprime1^2
                            x2^2*xprime2^2 + x1xprime1* x2*xprime2]
                            
But this is expensive so the alternative is:
K(x,xprime) = (1+x * xprime)^p where is P=2 X = 2
              (1+x1x*prime1+ ... xn * xprimen) ^ p where we could change to
                                                   P = 100 n = 15
             which can be finally converted to:
              (1,x1^2 * x2^2, sqrt(2)x1,sqrt(2)x2, sqrt(2)x1x2)
              
Other Kernal Radial Basis Function RBF
K (x1,xprime) = exp(-gamma * abs(x-xprime)^2)
        where   exp(x) = e^2
        
If the numberof supportvectors divided by the total num of samples
is more than 10%, then you might have issues.

Hard Margin Classifier: fits closely with classes
Soft Margin Classifier: Allows some degree of error or slack

So 
Slack = S where S >= 0
Slack = sum(S) however we want to minimize

Yi (xi * w + b) > 1 - S

so if we minimize:  ^2 + Csum(S)
So we want to minimize all this.
The more we raise C, the more strict we are
While if we lower the value, we are less strict
"""
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers


def linear_kernel(x, y):
    return np.dot(x, y)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


class SVM(object):
    def __init__(self, kernal=linear_kernel, C=None):
        self.kernel = kernal
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # support vactors have non lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b += np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        if self.kernel == linear_kernel:
            self.w = np.zeros((n_features))
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))


if __name__ == "__main__":
    import pylab as pl

    def gen_lin_separable_data():
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_non_lin_separable_data():
        mean1 = [-1,2]
        mean2 = [1,-1]
        mean3 = [4,-4]
        mean4 = [-4,4]
        cov = [[1.0,0.8],[0.8,1.0]]
        X1 = np.random.multivariate_normal(mean1, cov,50)
        X1 = np.vstack((X1,np.random.multivariate_normal(mean3,cov,50)))
        y1 = np.ones(len(X1))

        X2 = np.random.multivariate_normal(mean2, cov,50)
        X2 = np.vstack((X2,np.random.multivariate_normal(mean4,cov,50)))
        y2 = np.ones(len(X2)) * -1
        return X1,y1,X2,y2

    def gen_lin_separable_overlap_data():
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1,y1,X2,y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train,X2_train))
        y_train = np.hstack((y1_train,y2_train))
        return X1_train, y_train

    def split_test(X1,y1,X2,y2):
        X1_test = X1[90:]
        y1_test =y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test,X2_test))
        y_test = np.hstack((y1_test,y2_test))
        return X_test,y_test

    def plot_margin(X1_train,X2_train,clf):
        def f (x,w,b,c=0):
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:,0],X1_train[:,1],"ro")
        pl.plot(X2_train[:,0], X2_train[:,1],"bo")
        pl.scatter(clf.sv[:,0],clf.sv[:,1],s=100,c="g")

        # w.x +b = 0
        a0 = -4; a1 = f (a0,clf.w,clf.b)
        b0 = 4; b1 = f(b0,clf.w,clf.b)
        pl.plot([a0,b0], [a1,b1],"k")

        # w.x + b = 1
        a0 = -4; a1 = f (a0,clf.w,clf.b,1)
        b0 = 4; b1 = f(b0,clf.w,clf.b,1)
        pl.plot([a0,b0], [a1,b1],"k--")

        # w.x + b = -1
        a0 = -4; a1 = f (a0,clf.w,clf.b,-1)
        b0 = 4; b1 = f(b0,clf.w,clf.b,-1)
        pl.plot([a0,b0], [a1,b1],"k--")

