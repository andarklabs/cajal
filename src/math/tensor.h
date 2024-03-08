#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <string>
#include <cstring>
#include <iomanip> // for set_precision
#include <math.h>
#include <sstream> // for stringstream

class Tensor {

public:
  // initialize the tensor
  Tensor(unsigned int rows, unsigned int cols, float* data = nullptr);

  // return the tensor
  Tensor copy();

  // allow access to data
  unsigned int cols() const;
  unsigned int rows() const;
  // q: what float* not *data? 
  // a: float* is the type and data is the function name. 
  // float* means that it is a memory location that holds a float 
  // (a pointer to a float)
  float* data() const;
  std::string toString();

  static Tensor zeros(unsigned int rows, unsigned int cols);

  // static method

private:
  float* m_data;
  unsigned int m_rows;
  unsigned int m_cols;
};
#endif