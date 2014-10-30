# Something weird happens when you run this code.
# Find something that is not quite right.
# Figure out which compilation modes make the problem more obvious.
# Explain why what is happening is happening.
import numpy as np
from theano import function
from theano import tensor as T
x = T.vector()
y = T.vector()
z = T.zeros_like(y)
a = x + z
f = function([x, y], a, 'FAST_RUN')
output = f(np.zeros((1,), dtype=x.dtype), np.zeros((1,), dtype=y.dtype))
#The reason is the dimensions of x_input and y_input are not the same.
#With FAST_COMPILE mode, the problem will be revealed faster than the rest modes.
