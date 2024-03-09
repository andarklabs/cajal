// header of andark implementation of basic mat mult library

//TODO: follow naming conventions - https://www.geeksforgeeks.org/naming-convention-in-c/

#ifndef MAT_MATH_H
#define MAT_MATH_H

#include <iostream>

// aliases
using Matrix = float*;
using mat_size = size_t;
using shape = unsigned int*; // [dims, total_information, size_of_d_1, ... , size_of_d_dims]. 

// prints out contents of matrix. rows = r; columns = c.
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

// basic array multiplication. Varible names changed to reflect conceptual difference
Matrix arr_mult(float* A, float* B, size_t n);

// creating a matrix full of float z of size n * m if m = -1 then n*n is size
Matrix mat_init(float z, mat_size n, mat_size m = -1);

#endif