#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <iostream>
#include <chrono>


using Matrix = float*;

//TODO: int activate(string fn, Matrix m, unsigned int ti, float alpha = 0.0); /* This is a funtion that finds and calls your function that's name is your string */ 

void Relu(Matrix m, unsigned int ti);

void Lrelu(Matrix m, unsigned int ti, float alpha = 0.3);

void Softmax(Matrix m, unsigned int ti);

void Tanh(Matrix m, unsigned int ti);

void Sigmoid(Matrix m, unsigned int ti);

#endif