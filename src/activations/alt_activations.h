#ifndef ACTIVATIONS_H_a
#define ACTIVATIONS_H_a


#include <iostream>


using Matrix = float*;

void relu(Matrix m, unsigned int ti);

void lrelu(Matrix m, unsigned int ti);

void softmax(Matrix m, unsigned int ti);

void tanh(Matrix m, unsigned int ti);

#endif