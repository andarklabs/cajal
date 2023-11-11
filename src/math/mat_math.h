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

// aliases
using Tensor = float*;
using mat_size = size_t;


Tensor createMatrix(mat_size N, mat_size M = NULL){

    // Check if we are working with square matrices
    if (M == NULL){
        M = N;
    };

    // not value initalized. 
    Tensor C = new float[M*N];

    return C;

}

Tensor add(Tensor A, Tensor B, mat_size N){

}

Tensor sub(Tensor A, Tensor B, mat_size N){

}


Tensor naive_mult(Tensor A, Tensor B, mat_size n, mat_size m, mat_size p) {
    Tensor C = createMatrix(n,p);
    for (int i = 0; i<n ; i++){
        for (int j = 0; j<m ; j++){
            for (int k = 0; k<p ; k++){

            }
        }
    }

}


// divide and conquer on square strassen 
Tensor square_strassen(Tensor A, Tensor B, mat_size N){
// TODO: Fill in with square strassen implementation

    Tensor C = createMatrix(N);

    // this is our base case i'm implored to understand
	if (N == 1) {

        C[0] = A[0] * B[0];

		return C;
	}

    // sub matrix size 
    mat_size K = N/2;

    // we should create our sub matrices i guess
    Tensor A11 = createMatrix(K);
    Tensor A12 = createMatrix(K);
    Tensor A21 = createMatrix(K);
    Tensor A22 = createMatrix(K);
    Tensor B11 = createMatrix(K);
    Tensor B12 = createMatrix(K);
    Tensor B21 = createMatrix(K);
    Tensor B22 = createMatrix(K);

    // filling in our matrices
    for (unsigned int i = 0; i < K; i++) {
            for (unsigned j = 0; j < K; j++) {
                A11[i * K + j] = A[i * K + j];
                A12[i * K + j] = A[(i * K) + K + j];		
                A21[i * K + j] = A[((K + i) * K) + j];
                A22[i * K + j] = A[((K + i) * K) + K + j];
                B11[i * K + j] = B[i * K + j];
                B12[i * K + j] = B[(i * K) + K + j];		
                B21[i * K + j] = B[((K + i) * K) + j];
                B22[i * K + j] = B[((K + i) * K) + K + j];
            }
	}
    
    // S
	Tensor S1 = sub(B12, B22, K);
	Tensor S2 = add(A11, A12, K);
	Tensor S3 = add(A21, A22, K);
	Tensor S4 = sub(B21, B11, K);
	Tensor S5 = add(A11, A22, K);
	Tensor S6 = add(B11, B22, K);
	Tensor S7 = sub(A12, A22, K);
	Tensor S8 = add(B21, B22, K);
	Tensor S9 = sub(A11, A21, K);
	Tensor S10 = add(B11, B12, K);

	// P
	Tensor P1 = square_strassen(A11, S1, K);
	Tensor P2 = square_strassen(S2, B22, K);
	Tensor P3 = square_strassen(S3, B11, K);
	Tensor P4 = square_strassen(A22, S4, K);
	Tensor P5 = square_strassen(S5, S6, K);
	Tensor P6 = square_strassen(S7, S8, K);
	Tensor P7 = square_strassen(S9, S10, K);

	// C submatrices
	Tensor C11 = sub(add(add(P5, P4, K), P6, K), P2, K);				// P5 + P4 - P2 + P6
	Tensor C12 = add(P1, P2, K);								        // P1 + P2
	Tensor C21 = add(P3, P4, K);								        // P3 + P4
	Tensor C22 = sub(sub(add(P5, P1, K), P3, K), P7, K);				// P1 + P5 - P3 - P7

	// build our C matrix										
	for (unsigned int i = 0; i < K; i++) {
		for (unsigned int j = 0; j < K; j++) {

			C[i * K + j] = C11[i * K + j];

			C[(i * K) + K + j] = C12[i * K + j];

			C[((K + i) * K) + j] = C21[i * K + j];

			C[((K + i) * K) + K + j] = C22[i * K + j];
		}
	}

	// Return this matrix
	return C;


}

#endif