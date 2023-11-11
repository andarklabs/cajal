#ifndef MAT_MATH_H
#define MAT_MATH_H

#include <iostream>
#include "math/tensor.h"

using namespace std;


// maybe we can employ this somehow:
// template<int rows, int columns> using Matrix = int[rows*columns] *;

// for now we use this
using Matrix = int * *;
using mat_size = std::size_t;


Matrix createMatrix(mat_size N){

}

// divide and conquer on square strassen
Matrix square_strassen(Matrix A, Matrix B, mat_size N){
// TODO: Fill in with square strassen implementation

    Matrix C = createMatrix(N);

    // this is our base case
	if (N == 1) {

        C[0][0] = A[0][0] * B[0][0];
        
		return C;
	}

}

#endif