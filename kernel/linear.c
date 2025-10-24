#include "linear.h"

float *linear(float *input, float **weights, float *biases, int inputSize, int outputSize)
{
    // TODO: Implement the linear function (fully connected layer)
    float *output = (float *)malloc((size_t)outputSize * sizeof(float));
    if (!output) { return NULL; }

    for (int o = 0; o < outputSize; ++o)
    {
        float elem = 0;
        float *wrow = weights[o]; // array of floats from the row of weights, with length inputSize
        for (int i = 0; i < inputSize; ++i)
       	{
            elem += wrow[i] * input[i]; // computes Wx for a given out_i
        }
        output[o] = elem + biases[o];
    }
    return output;
}
