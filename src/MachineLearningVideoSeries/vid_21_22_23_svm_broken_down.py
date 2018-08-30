"""
A vector is a inside a vector space (or feature space)
So if we have a vector A = [3,4] and it has a maginitude
and direction

The direction and magnitude can be found.
Magnitude: sqrt(3^2 + 4^2)
Dot product is the multiplication of 2 vectors by their
elements (linear algebra)

So the vector machine does this:
barU is the unknown vector
barW is the vector from 0.0 to the hyperplane
b is the bias

So: barU*barW + b > 0 is positive
All that this equation is is if the vector is beyond the
hyperplane, it is positive. Also I can see how this is easily scalable to
other dimensions. The converse is negative.

If these equal 0, then then the element honestly cant be
determined.

So normally we know the values of barU, but we need the SVM
to find the values of both barW (hyperplane) and b (bias)

+Class is barXi * barW + b = 1
Yi = 1
-Class is barXi * barW + b = -1
Yi = -1

We can dirive the equation like so:
+Class is Yi(barXi * barW + b) = Yi(which is 1)(1)
Yi = 1
-Class is Yi(barXi * barW + b) = Yi(which is -1)(1)
Yi = -1

Which you get te same equation. We set each one to zero by
subtracting 1: Yi(barXi * barW + b) - 1 = 0

From my understanding, the point of this equation is to find the
hyperplane. That is all. If a vector causes this to be zero
then we have found a support vector, which can be used to
determine the hyperplane.

Support vector: affects the hyperplane for best separation

The best hyperplane can be found by finding the avg of the widths
between hyperplanes. So how do we do this.

Width = (Xpos -Xneg) (vecW / magW)
The goal is to maximize this width
width = (Yi(Xi * vecW + b) - 1) - (Yi(Xi * vecW + b) - 1)) * (vecW / magW)

which some how comes out to:
width = 2 / magW

Which since we want to maximize the width, we want to minimize magW
and for some reason we change this to width = (1/2) magW^2

So we want to minimize (1/2) (magW ^ 2) (Yi (Xi * vecW + b) - 1)
which is a constraint.

We use something called a lagrange multiplier which
is used for finding the max or min of a function with a constraint

L (w,b) = (1/2) (magW^2) - sum(alphai (Yi(vecXi * vecW + b)-1))

We want to minimize W and max b which increases the width.
So we need to find the derivative of L/W which is vecW - sum(yi *vecXi)
find the dirivative L/b = sum(alphai * yi) = 0

L = sum (alphai - (1/2sum(alphai * alphaj * yi * yj (vecXi * vecXj))))
We want to maximize this. This is a quadratic programming problem.
"""
