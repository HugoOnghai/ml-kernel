#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../kernel/linear.h"

int main() {
    srand(time(NULL));
    int inputSize = 65000;
    int outputSize = 65000;

    // generate random input, weights, and biases
    // input
    float *input = (float *)malloc((size_t)inputSize * sizeof(float));
    for (int i = 0; i < inputSize; i++) {
        input[i] = (float)rand() / RAND_MAX;
    }

    // weights
    float **weights = (float **)malloc((size_t)outputSize * sizeof(float *));
    for (int i = 0; i < outputSize; i++) {
        weights[i] = (float *)malloc((size_t)inputSize * sizeof(float));
        for (int j = 0; j < inputSize; j++) {
            weights[i][j] = (float)rand() / RAND_MAX;
        }
    }

    // biases
    float *biases = (float *)malloc((size_t)outputSize * sizeof(float));
    for (int i = 0; i < outputSize; i++) {
        biases[i] = (float)rand() / RAND_MAX;
    }

    // printf("Performing linear transformation...\n");
    float *output = linear(input, weights, biases, inputSize, outputSize);

    // Cleanup
    free(input);
    for (int i = 0; i < outputSize; i++) {
        free(weights[i]);
    }
    free(weights);
    free(biases);
    free(output);

    // printf("Linear transformation completed.\n");

    return 0;
}