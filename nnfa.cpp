// This program illustrates the universal approximation theorem for neural networks.
// Any (continuous) function on a compact interval can be approximated arbitrarily well by a neural network, with sigmoidal activation and having one input, some number of hidden nodes, and one output, where the output does not pass through the activation function.
// Clearly, better approximations require a larger number of nodes in the hidden layer.

#include "bitmap_image.hpp" // We need this for drawing and saving bitmaps.
// It is provided under the following agreement: https://opensource.org/licenses/cpl1.0.php

#include <cmath> // We need exp to define the sigmoid function.
#include <cstdlib> // We need rand and srand.
#include <ctime> // We need time.

#include <algorithm> // We need find.
#include <iostream> // We will use console input and output.
#include <vector> // We will use the class vector to store training data and plotting data.

using namespace std; // I want to avoid writing "std::" over and over.

// The function randomDouble generates a random double in the interval [a, b].
double randomDouble(const double &a = -1, const double &b = 1)
{
 return (static_cast<double>(rand()) / RAND_MAX) * (b - a) + a;
}

// We use the sigmoid function as our activation function.
double sigmoid(const double &a)
{
 return 1 / (1 + exp(-a));
}

// The class nnfa defines a feed-forward neural network.
// It consists of one input, n nodes in the hidden layer, and one output.
class nnfa {
 protected:
  unsigned int n; // This is the number of nodes in the hidden layer.
  double o_in; // This is the input.
  double *o_hidden; // These are the activation values of the hidden layer.

  double **w_hidden;
  // These are the weights in the hidden layer.
  // Given an index i, w_hidden[i] is an array of the weights for the i-th node in the hidden layer.
  // The last weight in the array w_hidden[i] is the bias for the i-th node.

  double *d_hidden;
  // These are used in adjusting the weights in the hidden layer.
  // They are calculated according to the backpropogation algorithm.

  double o_out; // This is the output.

  double *w_out;
  // These are the weights for the single node in the output layer.
  // The last weight in the array w_out is the bias.

  double d_out; // This is used in adjusting the weights in the output layer.
 public:

  // This constructs a neural network with a nodes in the hidden layer.
  // Initialize the network with random weights.
  explicit nnfa(const unsigned int &a)
  {
   unsigned int i;
   n = a;

   o_hidden = new double [n];
   w_hidden = new double *[n];
   for (i = 0; i < n; i ++)
   {
    w_hidden[i] = new double [2];
    w_hidden[i][0] = randomDouble();
    w_hidden[i][1] = randomDouble();
   }

   d_hidden = new double [n];

   w_out = new double [n + 1];
   for (i = 0; i <= n; i ++)
   {
    w_out[i] = randomDouble();
   }
  }

  // This is a standard destructor: it deletes what the constructor created.
  ~nnfa()
  {
   unsigned int i;
   delete[] w_out;
   delete[] d_hidden;
   for (i = 0; i < n; i ++)
   {
    delete[] w_hidden[i];
   }
   delete[] w_hidden;
   delete[] o_hidden;
  }

  // This computes the output of the neural network, given the input a.
  // It records the values of intermediate computations in some of the member variables.
  // These will allow us to do backpropogation afterward, if we want.
  double run(const double &a)
  {
   unsigned int i;
   o_in = a;
   for (i = 0; i < n; i ++)
   {
    o_hidden[i] = sigmoid(w_hidden[i][0] * o_in + w_hidden[i][1]);
   }
   o_out = w_out[n];
   for (i = 0; i < n; i ++)
   {
    o_out += w_out[i] * o_hidden[i];
   }

//   o_out = sigmoid(o_out);
// This program is designed to illustrate the universal approximation theorem.
// We do not pass the final output through the activation function.

   return o_out;
  }

  // This trains the neural network, where a is an input, and t is the correct output.
  // We compute the output of the neural network using run, and we correct the weights using backpropogation.
  void train(const double &a, const double &t)
  {
   const double eta = 0.05;
   unsigned int i;

   run(a);

//   d_out = o_out * (1 - o_out) * (t - o_out);
// We are illustrating the universal approximation theorem, where the output does not pass through the activation function.

   d_out = t - o_out; // This is the correct "adjustment" when we do not pass the output through the activation function.

   // What follows is the standard backpropogation algorithm.

   for (i = 0; i < n; i ++)
   {
    d_hidden[i] = o_hidden[i] * (1 - o_hidden[i]) * d_out * w_out[i];
   }

   w_out[n] += eta * d_out;
   for (i = 0; i < n; i ++)
   {
    w_out[i] += eta * d_out * o_hidden[i];
   }

   for (i = 0; i < n; i ++)
   {
    w_hidden[i][0] += eta * d_hidden[i] * o_in;
    w_hidden[i][1] += eta * d_hidden[i];
   }

   return;
  }
};

/*
// This function passes Octave code, illustrating the outputs of the neural network, to cout.
// We generate n random data points (i.e., x values) in the interval [a, b].
// We use the neural network to compute the corresponding y values.
void printOctave(nnfa &nn, const double &a, const double &b, const unsigned int &n = 200)
{
 vector<double> x, y;
 unsigned int i;
 for (i = 0; i < n; i ++)
 {
  x.push_back(randomDouble(a, b));
  y.push_back(nn.run(x.back()));
 }
 cout << "X = [" << x[0];
 for (i = 1; i < n; i ++)
 {
  cout << ", " << x[i];
 }
 cout << "];" << endl
      << "Y = [" << y[0];
 for (i = 1; i < n; i ++)
 {
  cout << ", " << y[i];
 }
 cout << "];" << endl
      << "plot(X, Y, \'.\');" << endl;
 return; 
}
*/

// The function plotfn draws the graphs of the functions in the vector f to a bitmap.
// The bitmap is saved under the name fname.
// The functions should be defined over the domain [x1, x2).
// The width of the bitmap is width.
// The height of the bitmap is calculated according to the equation width/(x2-x1)=height/(y2-y1).
void plotfn(
 const char *fname,
 double (*func)(const double &), nnfa &nn,
 const double &x1, const double &x2,
 const unsigned int &width
) {
 const double x_step = (x2 - x1) / width;

 vector<unsigned int> y_func;
 vector<unsigned int> y_nn;

 double y1 = 0;
 double y2 = 0;

 unsigned int i;
 int j;

 double x;
 double y;

 // Compute the "max" and "min" of func and nn.run.
 x = x1;
 for (i = 0; i < width; i ++)
 {
  y = func(x);
  if (y < y1)
  {
   y1 = y;
  }
  else if (y > y2)
  {
   y2 = y;
  }

  y = nn.run(x);
  if (y < y1)
  {
   y1 = y;
  }
  else if (y > y2)
  {
   y2 = y;
  }
  x += x_step;
 }

 cout << "Drawing plots on [" << x1 << ", " << x2 << "]x[" << y1 << ", " << y2 << "]..." << endl;

 y1 -= 0.1;
 y2 += 0.1;

 const unsigned int height = static_cast<unsigned int>(round((y2 - y1) * width / (x2 - x1)));
 // Compute a height adequate for displaying the graphs of both func and the neural network.

 const double y_step = (y2 - y1) / height;

 x = x1;
 for (i = 0; i < width; i ++)
 {
  y = func(x);
  j = height - 1 - static_cast<int>(round((y - y1) / y_step)); // Compute the height index corresponding to the coordinate y.
  y_func.push_back(j);

  y = nn.run(x);
  j = height - 1 - static_cast<int>(round((y - y1) / y_step)); // Compute the height index corresponding to the coordinate y.
  y_nn.push_back(j);

  x += x_step;
 }

 bitmap_image image(width, height);
 image.set_all_channels(255, 255, 255);
 // The bitmap now consists of all white pixels.

 image_drawer draw(image);
 draw.pen_width(2);

 // Plot the graph of func on [x1, x2).
 draw.pen_color(0, 0, 255);
 for (i = 1; i < width; i ++)
 {
  draw.line_segment(i - 1, y_func[i - 1], i, y_func[i]);
 }

 // Plot the graph of nn.run on [x1, x2).
 draw.pen_color(255, 0, 0);
 for (i = 1; i < width; i ++)
 {
  draw.line_segment(i - 1, y_nn[i - 1], i, y_nn[i]);
 }

 image.save_image(fname);

 cout << "The target function is drawn in blue, and the neural network is drawn in red." << endl;
 return;
}

// This function tries to teach a neural network the function func.
// The function func should be defined over the interval [a, b].
// The neural network consists of one input, N hidden nodes, and one output.
// The training set consists of T randomly chosen pairs (x, y=func(x)).
// We train the neural network over the whole training set C times.
void test(double (*func)(const double &), const double &a, const double &b, const unsigned int &N, const unsigned int &T, const unsigned int &C, const char *fname)
{
 nnfa nn(N);
 unsigned int i, j;
 vector<double> X, t;

// build training set

 cout << "Number of nodes in hidden layer: " << N << endl
      << "Size of training set: " << T << endl
      << "Number of times to train: " << C << endl;

 // random training set
 while (X.size() < T)
 {
  double temp = randomDouble(a, b);

  // We will only add this training value if it has not already been added.
  if (find(X.begin(), X.end(), temp) == X.end())
  {
   X.push_back(randomDouble(a,b));
   t.push_back(func(X.back()));
  }
 }
/*
 //
 double step = (b - a) / T;
 double point = a;
 for (i = 0; i < T; i ++)
 {
  X.push_back(point);
  t.push_back(func(X.back()));
  point += step;
 }*/

 for (i = 0; i < C; i ++)
 {
  // Pass over the training set one time.
  for (j = 0; j < X.size(); j ++)
  {
   nn.train(X[j], t[j]);
  }
  if ((i + 1) % (C / 5) == 0) cout << "Completed training #" << i + 1 << endl; // Let the user know our progress.
 }

 // After we finish training, we compute the error (i.e., sum of squared errors) of the neural network over the training set.
 double error = 0;
 for (i = 0; i < X.size(); i ++)
 {
  error += (nn.run(X[i]) - t[i]) * (nn.run(X[i]) - t[i]);
 }
 error *= 0.5;

 cout << "Total error over the training set: " << error << endl;

// printOctave(nn, a, b);

 plotfn(fname, func, nn, a, b, 500);

 return;
}

// This the function we teach our neural network.
double f(const double &x)
{
 const double PI = 3.1415926535897932384626;
// return sin(2 * PI * x) + exp(x) + sqrt(x) + (x + 2) * (x + 2) * sin(2 * PI * x);
// return 0.2+0.4*x*x+0.3*x*sin(15*x)+0.05*cos(50*x);
 return exp(x) * sin(PI * x);
}

int main()
{
 srand(time(0)); // Seed the random number generator.

 // Teach the neural network the function f on the interval [-1, 2].
 // Use a neural network with 10 nodes in the hidden layer.
 // Use a randomly generated training set with 1000 pairs (x, f(x)).
 // Train the neural network over the whole training set 50 times.
 test(f, -1, 2, 10, 1000, 50, "nnfa.bmp");

 return 0;
}
