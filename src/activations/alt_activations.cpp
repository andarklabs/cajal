#include "alt_activations.h"
#include "mat_math.h"

using Matrix = float*;

using namespace std;

// Rectified linear unit on 2d matrix
void relu(Matrix m, unsigned int ti) {

  for (int i = 0; i < ti; i++) {
      if (m[i] < 0) {
        m[i] = 0;
      }
    }
}

int main() {
    unsigned int s[] = {2,6,2,3}; // 2d 6ti 2r 3c
    float T[] = {4,5,7,8,10,12}; // [[4,5,7],[8,10,12]]
    Matrix test = T;
    relu(test, s[1]);
    toStr(test, s[2], s[3]); // defined in mat_math.cpp 
    return 0;
}