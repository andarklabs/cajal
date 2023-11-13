// some strassen code credit to psakoglou on github. 
// I altered where I could for optimization purposes.

// andark implementation of basic mat mult library

#include <math.h>
#include <iostream>
#include <cassert>
#include "mat_math.h"

using namespace std;

// create a tensor based off of the passed total information
Tensor createTensor(unsigned int ti, float data[] = NULL){
    
    // initalized to 0. ti is all dimension sizes multiplied together
    Tensor C = new float[ti]();
    
    // 
    if (data!=NULL){
        C = data;
    }

    return C;
}

// Add two Tensors together using their total informations
Tensor add(Tensor A, Tensor B, unsigned int ti, bool del = false){
    Tensor C = new float[ti];
    
    unsigned int total_info = ti;

    for (unsigned int i = 0; i < total_info; i++){
        C[i] = A[i] + B[i];
    }

    if (del){
        delete A;
        delete B;
    }

	return C;
}

// Subtract Tensor B from Tensor A using their total informations
Tensor sub(Tensor A, Tensor B, unsigned int ti, bool del = false){
    Tensor C = new float[ti];
    
    unsigned int total_info = ti;

    for (unsigned int i = 0; i < total_info; i++){
        C[i] = A[i] - B[i];
    }

    if (del){
        delete A;
        delete B;
    }

	return C;
}

// matrix A has n rows and p columns and matrix B has p rows and m columns. 
// The resultant matrix is n*m
Tensor naive_mult(Tensor A, Tensor B, mat_size n, mat_size m, mat_size p, bool del = false) {

    float sum = 0;

    Tensor C = new float[n*m]; // resultant matrix has n*m total information

    for (int i = 0; i<n ; i++){ // for row in A
        for (int j = 0; j<m ; j++){ // for column in B
            for (int k = 0; k<p ; k++){ // for column in A / row in B
                sum += A[(i * p) + k] * B[(k * m) + j];				
			}
			C[(i * m) + j] = sum;							
			sum = 0;							
		}
	}
    
    // free memory A and B
    delete A;
    delete B;

	return C;
}

// divide and conquer on square strassen 
Tensor square_strassen(Tensor A, Tensor B, mat_size n, bool del = false){

    Tensor C = new float[n];

    // this is our base case i'm implored to understand
	if (n == 1) {

        C[0] = A[0] * B[0];

		return C;
	}

    // sub Tensor size 
    mat_size k = n/2;

    // we should create our sub matrices i guess
    Tensor A11 = new float[k];
    Tensor A12 = new float[k];
    Tensor A21 = new float[k];
    Tensor A22 = new float[k];
    Tensor B11 = new float[k];
    Tensor B12 = new float[k];
    Tensor B21 = new float[k];
    Tensor B22 = new float[k];

    // filling in our matrices. We are using row-major order so C[i * N + j]
    // means the ith row and the jth column
    for (unsigned int i = 0; i < k; i++) {
            for (unsigned j = 0; j < k; j++) {

                A11[i * k + j] = A[i * k + j];
                A12[i * k + j] = A[(i * k) + k + j];		
                A21[i * k + j] = A[((k + i) * k) + j];
                A22[i * k + j] = A[((k + i) * k) + k + j];
                B11[i * k + j] = B[i * k + j];
                B12[i * k + j] = B[(i * k) + k + j];		
                B21[i * k + j] = B[((k + i) * k) + j];
                B22[i * k + j] = B[((k + i) * k) + k + j];
            }
	}
    
    // S
	Tensor S1 = sub(B12, B22, k);
	Tensor S2 = add(A11, A12, k);
	Tensor S3 = add(A21, A22, k);
	Tensor S4 = sub(B21, B11, k);
	Tensor S5 = add(A11, A22, k);
	Tensor S6 = add(B11, B22, k);
	Tensor S7 = sub(A12, A22, k);
	Tensor S8 = add(B21, B22, k);
	Tensor S9 = sub(A11, A21, k);
	Tensor S10 = add(B11, B12, k);

	// P
	Tensor P1 = square_strassen(A11, S1, k);
	Tensor P2 = square_strassen(S2, B22, k);
	Tensor P3 = square_strassen(S3, B11, k);
	Tensor P4 = square_strassen(A22, S4, k);
	Tensor P5 = square_strassen(S5, S6, k);
	Tensor P6 = square_strassen(S7, S8, k);
	Tensor P7 = square_strassen(S9, S10, k);

	// C submatrices
	Tensor C11 = sub(add(add(P5, P4, k), P6, k), P2, k);				// P5 + P4 - P2 + P6
	Tensor C12 = add(P1, P2, k);								        // P1 + P2
	Tensor C21 = add(P3, P4, k);								        // P3 + P4
	Tensor C22 = sub(sub(add(P5, P1, k), P3, k), P7, k);				// P1 + P5 - P3 - P7

	// build our C matrix										
	for (unsigned int i = 0; i < k; i++) {
		for (unsigned int j = 0; j < k; j++) {

			C[i * k + j] = C11[i * k + j];
			C[(i * k) + k + j] = C12[i * k + j];
			C[((k + i) * k) + j] = C21[i * k + j];
			C[((k + i) * k) + k + j] = C22[i * k + j];
		}
	}

    // free memory A and B
    delete A;
    delete B;
    delete P1; 
	delete P2;
	delete P3;
	delete P4;
	delete P5;	
    delete P6;	
    delete P7;
    delete S1; 
	delete S2;
	delete S3;
	delete S4;
	delete S5;	
    delete S6;	
    delete S7;
    delete S8;	
    delete S9;	
    delete S10;

	// Return this matrix
	return C;


}