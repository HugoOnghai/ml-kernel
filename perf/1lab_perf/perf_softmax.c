# include "../kernel/functional.h"
# include <stdio.h>

void main() {
    int inputSize = 1e2;

    float *input = (float *)calloc((size_t)inputSize, sizeof(float));
    for (int i = 0; i < inputSize; i++) {
        input[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f;
    }

    // printf("input: %f %f %f %f %f\n", input[0], input[1], input[2], input[3], input[4]);

    float *output = softmax(input, inputSize);

    // printf("output: %f %f %f %f %f\n", output[0], output[1], output[2], output[3], output[4]);

    free(input);
    free(output);
}