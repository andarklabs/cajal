#include "dense.hpp"
#include "initializers/initializers.hpp"
#include "math/mat_math.hpp"

using namespace std;

Dense::Dense(unsigned int numInputs, unsigned int numOutputs, std::string activationPassed){

    num_inputs = numInputs;
    num_outputs = numOutputs;
    num_weights = numInputs * numOutputs;
    activation = activationPassed;

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
    int n = 10;
    Dense layer1 = Dense(n,n, "relu");
    float inp[10] = {1,2,3,4,5,6,7,8,9,0};
    float* arr = layer1.ForwardProp(inp);

    ToStr(arr, 1, 10);

    return 0;
}