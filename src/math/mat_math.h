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

Matrix add(Matrix A, Matrix B){

}

Matrix sub(Matrix A, Matrix B){

}


Matrix naive_mult(Matrix A, Matrix B, mat_size n, mat_size m, mat_size k) {


}

// divide and conquer on square strassen 
Matrix square_strassen(Matrix A, Matrix B, mat_size N){
// TODO: Fill in with square strassen implementation

    Matrix C = createMatrix(N);

    // this is our base case i'm implored to understand
	if (N == 1) {

        C[0][0] = A[0][0] * B[0][0];

		return C;
	}

    // i wonder what size our sub matrices should be?
    mat_size K = N/2;

    // we should create our sub matrices i guess
    Matrix A11 = createMatrix(K);
    Matrix A12 = createMatrix(K);
    Matrix A21 = createMatrix(K);
    Matrix A22 = createMatrix(K);
    Matrix B11 = createMatrix(K);
    Matrix B12 = createMatrix(K);
    Matrix B21 = createMatrix(K);
    Matrix B22 = createMatrix(K);

    // filling in our matrices
    for (unsigned int i = 0; i < K; i++) {
            for (unsigned j = 0; j < K; j++) {
                A11[i][j] = A[i][j];
                A12[i][j] = A[i][K + j];		
                A21[i][j] = A[K + i][j];
                A22[i][j] = A[K + i][K + j];
                B11[i][j] = B[i][j];
                B12[i][j] = B[i][K + j];		
                B21[i][j] = B[K + i][j];
                B22[i][j] = B[K + i][K + j];
            }
	}



}

#endif