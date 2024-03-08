#ifndef ACTIVATIONS_H_a
#define ACTIVATIONS_H_a

#include <iostream>


using Matrix = float*;

//ptr activate(string fn); /* This */

void relu(Matrix m, unsigned int ti);

void lrelu(Matrix m, unsigned int ti, float alpha = 0.3);

void softmax(Matrix m, unsigned int ti);

void tanh(Matrix m, unsigned int ti);

void sigmoid(Matrix m, unsigned int ti);

#endif