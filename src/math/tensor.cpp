#include "tensor.h"
#include <cstring>
#include <iomanip> // for set_precision
#include <iostream>
#include <math.h>
#include <sstream> // for stringstream

using namespace std;

Tensor::Tensor(unsigned int rows, unsigned int cols, float *data) {
  // allocate heap mem for m_data
  m_data = new float[rows * cols]();

  if (data) {
    memcpy(m_data, data, rows * cols * sizeof(float));
    m_cols = cols;
    m_rows = rows;
  }
}

std::string Tensor::toString() {
  // build up a multiline string repr of the matrix
  std::stringstream ss;

  for (int i = 0; i < m_rows; i++) {
    for (int j = 0; j < m_cols; j++) {
      // at i,j, in mem its flat so get curr row* which col + offset
      ss << std::setprecision(3) << m_data[i * m_cols + j] << " ";
    }
    ss << std::endl;
  }
  return ss.str();
}

int main() {
  float data_a[] = {1.01, 2.11, 3.01, 4.01, 5.01, 6.01};
  Tensor A = Tensor(2, 3, data_a);
  cout << A.toString();
  return 0;
}