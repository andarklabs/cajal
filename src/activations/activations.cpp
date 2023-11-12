#include "activations.h"
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace std;

void relu(vector<vector<int>> &matrix) {
  /*
  -1 0 2
   1 2 0
   =====
   0 0 2
   1 2 0
  */
  for (int i = 0; i < matrix.size(); i++) {
    for (int j = 0; j < matrix[i].size(); j++) {
      if (matrix[i][j] < 0) {
        matrix[i][j] = 0;
      };
    };
  }
}

string toString(const vector<vector<int>> &m_data) {
  // build up a multiline string repr of the matrix
  std::stringstream ss;

  for (int i = 0; i < m_data.size(); i++) {
    for (int j = 0; j < m_data[i].size(); j++) {
      // at i,j, in mem its flat so get curr row* which col + offset
      ss << std::setprecision(3) << m_data[i][j] << " ";
    }
    ss << std::endl;
  }
  return ss.str();
}

int main() {
  vector<vector<int>> test = {{-1, 0, 2}, {1, 2, 0}};
  relu(test);
  cout << toString(test);
  return 0;
}