// header of andark implementation of basic mat mult library

#ifndef MAT_MATH_H
#define MAT_MATH_H

#include <iostream>
#include "math/tensor.h"

// aliases
using Matrix = float*;
using mat_size = size_t;
using shape = unsigned int *; // [dims, total_information, size_of_d_1, ... , size_of_d_dims]. 

void toStr(Matrix A, unsigned int r, unsigned int c);

// Add two Matrices together using their total informations
Matrix add(Matrix A, Matrix B, unsigned int ti, bool del = false);

// Subtract Matrix B from Matrix A using their total informations
Matrix sub(Matrix A, Matrix B, unsigned int ti, bool del = false);

// matrix A has n rows and p columns and matrix B has p rows and m columns. 
// The resultant matrix is n*m
Matrix naive_mult(Matrix A, Matrix B, mat_size n, mat_size m, mat_size p, bool del = false);

// divide and conquer on square strassen 
Matrix square_strassen(Matrix A, Matrix B, mat_size n, bool del = false);

#endif