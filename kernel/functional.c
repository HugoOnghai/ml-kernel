#include <math.h>
#include <float.h>
#include <stdlib.h>
#include "functional.h"

float relu(float x)
{
    // TODO: Implement relu
    return fmax(0, x);
}

void applyRelu(float *input, int inputSize)
{
    for (int i = 0; i < inputSize; i++)
    {
        input[i] = relu(input[i]);
    }
}

float *softmax(float *input, int inputSize)
{
    // TODO: Implement softmax
    if (!input || inputSize <= 0) return NULL;

    // Find maximum of input vector
    float maxInput = -FLT_MAX; // minimum finite float value, cppreference
    for (int i = 0; i < inputSize; i++)
    {
	if (input[i] > maxInput) maxInput = input[i];
    }

    float *out = (float *)malloc(sizeof(float) * (size_t)inputSize);
    if (!out) return NULL;

    // Compute exp of input - maxInput to avoid underflow
    for (int i = 0; i < inputSize; i++)
    {
	out[i] = expf(input[i] - maxInput); // using expf for float exponentiation
    }

    // Normalise and apply log
    float sum = 0;
    for (int i = 0; i < inputSize; i++)
    {
	sum += out[i];
    }
    
    for (int i = 0; i < inputSize; i++)
    {
	out[i] = logf(out[i] / sum); // using logf for float cppreference
    }

    return out;
}
