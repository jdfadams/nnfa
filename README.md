nnfa.cpp - This file contains the main function and all of my code.

bitmap_image.hpp - This file allows manipulation of bitmap files. I am not the author, but it's use is allowed under the agreement https://opensource.org/licenses/cpl1.0.php. See https://github.com/ArashPartow/bitmap for more information.

I compiled nnfa.cpp with g++ (Ubuntu 5.4.0-6ubuntu1~16.04.2) 5.4.0 20160609.

The purpose of this program is to illustrate the universal approximation theorem for neural networks in one of its most basic forms: Any continuous function on a compact interval (the target function) can be approximated arbitrarily well by a neural network (with a sigmoidal activation function) consisting of one input, some number of nodes in the hidden layer, and one output, where the output does not pass through the activation function. Of course, for a better approximation, one usually needs more nodes in the hidden layer.

The neural network itself is implemented in the class nnfa.

There is a function called test. This function trains the neural network using randomly chosen points in the domain interval.

There is a function called plotfn. This function draws the graph of the target function in blue and the graph of what the neural network has learned in red and saves the resulting bitmap.

See "Approximation by superpositions of a sigmoidal function" by G. Cybenko (http://link.springer.com/article/10.1007/BF02551274) for the relevant mathematics.
