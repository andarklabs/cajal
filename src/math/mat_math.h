// some strassen code credit to psakoglou on github. 
// I altered where I could for optimization purposes.

// andark implementation of basic mat mult library

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


Matrix createMatrix(mat_size N, mat_size M = NULL){

    // Check if we are working with square matrices
    if (M == NULL){
        M = N;
    }



}

Matrix add(Matrix A, Matrix B, mat_size N){

}

Matrix sub(Matrix A, Matrix B, mat_size N){

}


Matrix naive_mult(Matrix A, Matrix B, mat_size n, mat_size m, mat_size p) {
    Matrix C = createMatrix(n,p);
    for (int i = 0; i<n ; i++){
        for (int j = 0; j<m ; j++){
            for (int k = 0; k<p ; k++){

            }
        }
    }

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

    // sub matrix size 
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
    
    // S
	Matrix S1 = sub(B12, B22, K);
	Matrix S2 = add(A11, A12, K);
	Matrix S3 = add(A21, A22, K);
	Matrix S4 = sub(B21, B11, K);
	Matrix S5 = add(A11, A22, K);
	Matrix S6 = add(B11, B22, K);
	Matrix S7 = sub(A12, A22, K);
	Matrix S8 = add(B21, B22, K);
	Matrix S9 = sub(A11, A21, K);
	Matrix S10 = add(B11, B12, K);

	// P
	Matrix P1 = square_strassen(A11, S1, K);
	Matrix P2 = square_strassen(S2, B22, K);
	Matrix P3 = square_strassen(S3, B11, K);
	Matrix P4 = square_strassen(A22, S4, K);
	Matrix P5 = square_strassen(S5, S6, K);
	Matrix P6 = square_strassen(S7, S8, K);
	Matrix P7 = square_strassen(S9, S10, K);

	// C submatrices
	Matrix C11 = sub(add(add(P5, P4, K), P6, K), P2, K);				// P5 + P4 - P2 + P6
	Matrix C12 = add(P1, P2, K);								        // P1 + P2
	Matrix C21 = add(P3, P4, K);								        // P3 + P4
	Matrix C22 = sub(sub(add(P5, P1, K), P3, K), P7, K);				// P1 + P5 - P3 - P7

	// build our C matrix										
	for (unsigned int i = 0; i < K; i++) {
		for (unsigned int j = 0; j < K; j++) {

			C[i][j] = C11[i][j];

			C[i][j + K] = C12[i][j];

			C[K + i][j] = C21[i][j];

			C[K + i][K + j] = C22[i][j];
		}
	}

	// Return this matrix
	return C;


}

#endif