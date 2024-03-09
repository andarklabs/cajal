#ifndef DENSE_H
#define DENSE_H

#include <iostream>
#include <stack>
#include <vector>

typedef float Scalar;

class Dense{

public:
    // Constructor creates inital weights and biases for one layer
    Dense(int num_perceptrons, int num_inputs); 
    
    ~Dense(); // Destuctor

    void cleanLayer(); 
    int* forwardProp(); // returns an integer array representing the layer
    void backwardProp();


    
private:

};
#endif