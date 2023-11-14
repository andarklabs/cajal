#ifndef DENSE_H
#define DENSE_H

#include <iostream>
#include <stack>
#include <vector>
#include "perceptrons.h"

typedef float Scalar;

class Dense{

public:
    Dense(int num_perceptrons); // Constructor
    ~Dense(); // Destuctor

    void cleanLayer(); 
    void forwardProp();
    void backwardProp();


    
private:

};
#endif