#ifndef INITIALIZERS_H
#define INITIALIZERS_H

#include <iostream>
#include <random>
#include <string>
#include <functional>
#include <cmath>

float* InitWeights(unsigned int inp, unsigned int outp, const std::string technique = "xavier", const std::string distribution = "uniform");

std::function<float()> Xavier(unsigned int inp, unsigned int outp, const std::string distribution = "uniform");

std::function<float()> He(unsigned int inp, const std::string distribution = "uniform");

#endif