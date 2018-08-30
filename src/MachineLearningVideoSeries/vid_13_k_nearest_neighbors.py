"""
First part of classification. Starting with K nearest neighbors.

We need ot create a model that divides negative elements and
positive elements. While classifying elements that are close to 
some other points is easy, we might have a point that is close to 
the middle, OR we might have a data set with several dimentions
(properties). This is where we go into having a machine perform
k nearest neighbors. 
 
So lets say K = 2. This means that we look at the 2 closest neighbors 
to a point. HOWEVER, you might want to use even numbers such as 3
so that you can avoid ties. 
 
You also want to avoid setting K to the number of groups. If there are 3 
groups, then K = 3 would be undesirable, so maybe use K = 5. 
"""