// header of andark implementation of basic mat mult section of cajal library

//TODO: follow naming conventions - https://www.geeksforgeeks.org/naming-convention-in-c/
//TODO: change from floats to doubles

#ifndef MAT_MATH_H
#define MAT_MATH_H

#include <math.h>
#include <iostream>
#include <cassert>

// aliases
using Matrix = float*;
using mat_size = size_t;
using shape = unsigned int*; // [dims, total_information, size_of_d_1, ... , size_of_d_dims]. 

// creating a matrix full of float z of size n * m if m = -1 then n*n is size
Matrix MatInit(float z, mat_size n, mat_size m = -1);

// Add two Matrices together using their total informations
Matrix Add(Matrix A, Matrix B, unsigned int ti, bool del = false);

// Subtract Matrix B from Matrix A using their total informations
Matrix Sub(Matrix A, Matrix B, unsigned int ti, bool del = false);

// matrix A has n rows and p columns and matrix B has p rows and m columns. 
// The resultant matrix is n*m
Matrix NaiveMult(Matrix A, Matrix B, mat_size n, mat_size m, mat_size p, bool del = false);

// divide and conquer on square strassen 
Matrix SquareStrassen(Matrix A, Matrix B, mat_size n, bool del = false);

// basic array multiplication. Varible names changed to reflect conceptual difference
float* ArrMult(float* A, float* B, size_t n);

// basic dot product
float Dot(float* v, float* w, size_t n);

// prints out contents of matrix. rows = r; columns = c.
void ToStr(Matrix A, unsigned int r, unsigned int c);


#endif