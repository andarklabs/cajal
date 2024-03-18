#ifndef DENSE_H
#define DENSE_H

#include <iostream>
#include <stack>
#include <vector>
#include <string>
#include <math.h>

typedef float Scalar;

class Dense{

public:
    // Constructor creates inital weights and biases for one layer
    Dense(unsigned int numInputs, unsigned int numOutputs, std::string activationPassed); 
    
    ~Dense(); // Destuctor

    float* ForwardProp(float* inputs); // returns a float array representing the layer
    
    void BackwardProp(float loss);

private:

    float* weights;
    float* biases;
    unsigned int num_inputs;
    unsigned int num_outputs;
    unsigned int num_weights;
    std::string activation;

};
#endif