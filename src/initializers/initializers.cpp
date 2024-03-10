#include "initializers.hpp"
#include "mat_math.hpp"

using namespace std;

// all functions allocate a new array `C`. Remember to delete it.

float* InitWeights(unsigned int inp, unsigned int outp, const std::string technique /* = "xavier" */, const std::string distribution /* = "uniform" */){
    
    std::function<float()> rand_func;
    
    float* C = new float[inp*outp];

    if (technique == "xavier") {
        rand_func = Xavier(inp, outp, distribution);
    } else if (technique == "he"){
        rand_func = He(inp, distribution);
    } else {
        cout << "Error: invalid technique"<< std::endl;
    }

    for (int i  = 0; i < inp * outp; i++){
        C[i] = rand_func();
    }

    return C;
}

std::function<float()> Xavier(unsigned int inp, unsigned int outp, const std::string distribution /* = "uniform" */ ){ 

    std::random_device rd;
    std::mt19937 gen(rd());
    std::function<float()> rand_func;

    float r = 0.0;
    
    if (distribution == "uniform") {
        r = sqrt(6/(inp + outp));
        std::uniform_int_distribution<float> distrib(-r, r);
        rand_func = [&]() { return distrib(gen); };
    } else if (distribution == "normal") {
        r = sqrt(2/(inp + outp));
        std::normal_distribution<float> distrib(0,r);
        rand_func = [&]() { return distrib(gen); };
    } else {
        cout << "Error: invalid distribution" << std::endl;
    }
    
    return rand_func;

}

std::function<float()> He(unsigned int inp, const std::string distribution /* = "uniform" */ ){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::function<float()> rand_func;

    float r = 0.0;
    
    if (distribution == "uniform") {
        r = sqrt(6/(inp));
        std::uniform_int_distribution<float> distrib(-r, r);
        rand_func = [&]() { return distrib(gen); };
    } else if (distribution == "normal") {
        r = sqrt(2/(inp));
        std::normal_distribution<float> distrib(0,r);
        rand_func = [&]() { return distrib(gen); };
    } else {
        cout << "Error: invalid distribution" << std::endl;
    }
    
    return rand_func;

}