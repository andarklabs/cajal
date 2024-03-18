// some strassen code credit to psakoglou on github. 
// I altered where I could for optimization purposes.

// andark implementation of basic mat mult library

#include "mat_math.hpp"

using namespace std;

void ToStr(Matrix m, unsigned int r, unsigned int c) {

  for (int i = 0; i < r; i++) { // for num_dimensions
    for (int j = 0; j < c; j++) { // for size of each dimension
        cout<<m[(i * c) + j]<<" ";
    }
    cout<< '\n';
  }
}

// Add two Matrices together using their total informations
Matrix Add(Matrix A, Matrix B, unsigned int ti, bool del /* = false */){
    Matrix C = new float[ti];
    
    unsigned int total_info = ti;

    for (unsigned int i = 0; i < total_info; i++){
        C[i] = A[i] + B[i];
    }

    if (del){
        // free memory A and B
        delete[] A;
        delete[] B;
    }

	return C;
}

// Subtract Matrix B from Matrix A using their total informations
Matrix Sub(Matrix A, Matrix B, unsigned int ti, bool del /* = false */){
    Matrix C = new float[ti];
    
    unsigned int total_info = ti;

    for (unsigned int i = 0; i < total_info; i++){
        C[i] = A[i] - B[i];
    }

    if (del){
        // free memory A and B
        delete[] A;
        delete[] B;
    }

	return C;
}

// matrix A has n rows and p columns and matrix B has p rows and m columns. 
// The resultant matrix is n*m
Matrix NaiveMult(Matrix A, Matrix B, mat_size n, mat_size m, mat_size p, bool del /* = false */) {

    float sum = 0;

    Matrix C = new float[n*m]; // resultant matrix has n*m total information

    for (int i = 0; i<n ; i++){ // for row in A
        for (int j = 0; j<m ; j++){ // for column in B
            for (int k = 0; k<p ; k++){ // for column in A / row in B
                sum += A[(i * p) + k] * B[(k * m) + j];				
			}
			C[(i * m) + j] = sum;							
			sum = 0;							
		}
	}
    
    if (del){
        // free memory A and B
        delete[] A;
        delete[] B;
    }

	return C;
}

// divide and conquer on square strassen 
Matrix SquareStrassen(Matrix A, Matrix B, mat_size n, bool del /* = false */){

    Matrix C = new float[n];

    // this is our base case
	if (n == 1) {

        C[0] = A[0] * B[0];

		return C;
	}

    // Sub Matrix size 
    mat_size k = n/2;

    // we should create our Sub matrices 
    Matrix A11 = new float[k];
    Matrix A12 = new float[k];
    Matrix A21 = new float[k];
    Matrix A22 = new float[k];
    Matrix B11 = new float[k];
    Matrix B12 = new float[k];
    Matrix B21 = new float[k];
    Matrix B22 = new float[k];

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
	Matrix S1 = Sub(B12, B22, k);
	Matrix S2 = Add(A11, A12, k);
	Matrix S3 = Add(A21, A22, k);
	Matrix S4 = Sub(B21, B11, k);
	Matrix S5 = Add(A11, A22, k);
	Matrix S6 = Add(B11, B22, k);
	Matrix S7 = Sub(A12, A22, k);
	Matrix S8 = Add(B21, B22, k);
	Matrix S9 = Sub(A11, A21, k);
	Matrix S10 = Add(B11, B12, k);

	// P - also we can delete all S matrices right here 
    // along with A11, A22, B11, and B22
	Matrix P1 = SquareStrassen(A11, S1, k, true);
	Matrix P2 = SquareStrassen(S2, B22, k, true);
	Matrix P3 = SquareStrassen(S3, B11, k, true);
	Matrix P4 = SquareStrassen(A22, S4, k, true);
	Matrix P5 = SquareStrassen(S5, S6, k, true);
	Matrix P6 = SquareStrassen(S7, S8, k, true);
	Matrix P7 = SquareStrassen(S9, S10, k, true);

	// C Submatrices
	Matrix C11 = Sub(Add(Add(P5, P4, k), P6, k), P2, k);				// P5 + P4 - P2 + P6
	Matrix C12 = Add(P1, P2, k);								        // P1 + P2
	Matrix C21 = Add(P3, P4, k);								        // P3 + P4
	Matrix C22 = Sub(Sub(Add(P5, P1, k), P3, k), P7, k);				// P1 + P5 - P3 - P7

	// build our C matrix										
	for (unsigned int i = 0; i < k; i++) {
		for (unsigned int j = 0; j < k; j++) {

			C[i * k + j] = C11[i * k + j];
			C[(i * k) + k + j] = C12[i * k + j];
			C[((k + i) * k) + j] = C21[i * k + j];
			C[((k + i) * k) + k + j] = C22[i * k + j];
		}
	}

    // ----------- time to free memory ------------ //

    if (del){
        // free memory A and B
        delete[] A;
        delete[] B;
    }

    // delete[] remaining base matrices
    delete[] A12; 
	delete[] A21;
	delete[] B12;
	delete[] B21;

    // delete[] all P matrices
    delete[] P1; 
	delete[] P2;
	delete[] P3;
	delete[] P4;
	delete[] P5;	
    delete[] P6;	
    delete[] P7;

    // delete[] our C quadrants
    delete[] C11; 
	delete[] C12;
	delete[] C21;
	delete[] C22;

	// Return this matrix
	return C;


}

float* ArrMult(float* A, float* B, size_t n){

    float* C = new float[n]; // resultant arr has n total information

    for (unsigned int i = 0; i < n; i++) {
        C[i] = A[i] * B[i];
    }

    return C;
}

Matrix MatInit(float z, mat_size n, mat_size m /* = -1 */){

    if (m == -1) {
        float C[n];
        std::fill_n(C,n,z);
        return C;
    }

    else {
        float C[n*m];
        std::fill_n(C,n,z);
        return C;
    }

}

float Dot(float* v, float* w, size_t n){

    float product;
    
    for (unsigned int i = 0; i < n; i++){
        product += v[i] * w[i]; 
    }

    return product;
}