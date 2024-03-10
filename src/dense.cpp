#include "dense.hpp"
#include "initializers.hpp"
#include "mat_math.hpp"

using namespace std;

Dense::Dense(unsigned int num_inputs, unsigned int num_outputs, const std::string activation){

    num_inputs = num_inputs;
    num_outputs = num_outputs;
    num_weights = num_inputs * num_outputs;

    if (activation == "relu" || activation == "lrelu") {                // He init
        weights = InitWeights(num_inputs, num_outputs, "he");
    } else {                                                            // Xavier init
        weights = InitWeights(num_inputs, num_outputs);
    }

    biases = MatInit(0,num_outputs);

}

float* Dense::ForwardProp(float* inputs){

    unsigned int indx = 0;

    float outputs[num_outputs];

    for (unsigned int i = 0; i < num_outputs; i++){

        float sum = 0.0;
        for (unsigned int j = 0; j < num_inputs; j++){
            sum += inputs[j] * weights[(num_inputs*i) + j];
        }
        sum += biases[i];
        outputs[i] = sum;
    }

    return outputs;

}

int main() {
  unsigned int s[] = {2,6,2,3}; // 2d 6ti 2r 3c
  float test[] = {4,5,-7,8,10,12}; // [[4,5,7],[8,10,12]]
  Matrix M = MatInit(0,6);
  ToStr(M, s[2], s[3]);
  ToStr(test, s[2], s[3]); // defined in mat_math.cpp
  return 0;
}