#include "alt_activations.h"
#include "mat_math.h"

using Matrix = float*;

using namespace std;

// All these functions have inputs 'm' that are changed dynamically

// Rectified linear unit on vector
void relu(Matrix m, unsigned int ti) {

  for (int i = 0; i < ti; i++) {
      if (m[i] < 0) {
        m[i] = 0;
      }
    }
}

// Leaky rectified linear unit on vector
void lrelu(Matrix m, unsigned int ti, float alpha /* = 0.3 */) {

  for (int i = 0; i < ti; i++) {
      if (m[i] < 0) {
        m[i] *= alpha;
      }
    }
}

// 
void softmax(Matrix m, unsigned int ti){

  float mx = -INFINITY;
  for (unsigned int i=0; i<ti;i++){
    if (m[i] > mx){
      mx = m[i];
    }
  }

  float sum = 0.0;
  for (unsigned int i = 0; i < ti; i++){
    sum += expf(m[i] - mx);
  }

  float offset = mx + logf(sum);
  for (unsigned int i = 0; i < ti; i++){
    m[i] = expf(m[i] - offset);
  }

}

void tanh(Matrix m, unsigned int ti){
  for (unsigned int i = 0; i < ti; i++ ){
    m[i] = (2 / (1 + expf(-2 * m[i]) ) ) - 1;
  }
}

int main() {
  unsigned int s[] = {2,6,2,3}; // 2d 6ti 2r 3c
  float T[] = {4,5,-7,8,10,12}; // [[4,5,7],[8,10,12]]
  Matrix test = T;
  lrelu(test, s[1]);
  toStr(test, s[2], s[3]); // defined in mat_math.cpp 
  return 0;
}