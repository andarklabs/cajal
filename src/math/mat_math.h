// header of andark implementation of basic mat mult library

#ifndef MAT_MATH_H
#define MAT_MATH_H

#include <iostream>
#include "math/tensor.h"

using namespace std;

// aliases
using Tensor = float*;
using mat_size = size_t;
using shape = unsigned int*; // [dims, total_information, size_of_d_1, ... , size_of_d_dims]. 

// create a tensor based off of the passed total information
Tensor createTensor(unsigned int ti);

// Add two Tensors together using their total informations
Tensor add(Tensor A, Tensor B, unsigned int ti);

// Subtract Tensor B from Tensor A using their total informations
Tensor sub(Tensor A, Tensor B, unsigned int ti);

// matrix A has n rows and p columns and matrix B has p rows and m columns. 
// The resultant matrix is n*m
Tensor naive_mult(Tensor A, Tensor B, mat_size n, mat_size m, mat_size p);

// divide and conquer on square strassen 
Tensor square_strassen(Tensor A, Tensor B, mat_size n);

#endif