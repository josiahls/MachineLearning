"""
SEE:
   This breast cancer databases was obtained from the University of Wisconsin
   Hospitals, Madison from Dr. William H. Wolberg.  If you publish results
   when using this database, then please include this information in your
   acknowledgements.  Also, please cite one or more of:

   1. O. L. Mangasarian and W. H. Wolberg: "Cancer diagnosis via linear
      programming", SIAM News, Volume 23, Number 5, September 1990, pp 1 & 18.

   2. William H. Wolberg and O.L. Mangasarian: "Multisurface method of
      pattern separation for medical diagnosis applied to breast cytology",
      Proceedings of the National Academy of Sciences, U.S.A., Volume 87,
      December 1990, pp 9193-9196.

   3. O. L. Mangasarian, R. Setiono, and W.H. Wolberg: "Pattern recognition
      via linear programming: Theory and application to medical diagnosis",
      in: "Large-scale numerical optimization", Thomas F. Coleman and Yuying
      Li, editors, SIAM Publications, Philadelphia 1990, pp 22-30.

   4. K. P. Bennett & O. L. Mangasarian: "Robust linear programming
      discrimination of two linearly inseparable sets", Optimization Methods
      and Software 1, 1992, 23-34 (Gordon & Breach Science Publishers).

This script uses the above data to test k-nearest neighbors

It uses euclidean distance willbe used:

Is: sqrt ( sum n ((qi - Pi)^2) i = 1 )
where q is the first point 1,3
P is 2,5

So: sqrt ( (1 - 2)^2 + (3 - 5)^2  )
"""
from math import sqrt

plot1 = [1,3]
plot2 = [2,5]

euclid_distance = sqrt((plot1[0] - plot2[0])**2 + (plot1[1] -plot2[1])**2)

print(euclid_distance)





