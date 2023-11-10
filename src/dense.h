#ifndef DENSE_H
#define DENSE_H

#include <iostream>
#include <stack>
#include "perceptrons.h"

using namespace std;

class Dense{

public:
    Dense(int num_perceptrons); // Constructor
    ~Dense(); // Destuctor

    void cleanLayer(); 


    
private:

    stack<Perceptron*> layerStack;

};
#endif