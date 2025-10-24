#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "test_linear.h"


void test_linear_basic(void)
{
    float input[] = {1.0, 2.0, 3.0};
    float *weights[] = {(float[]){1.0, 2.0, 3.0}, (float[]){4.0, 5.0, 6.0}};
    float biases[] = {0.1, 0.2};
    float *output = linear(input, weights, biases, 3, 2);
    TEST_ASSERT_EQUAL_FLOAT(14.1, output[0]);
    TEST_ASSERT_EQUAL_FLOAT(32.2, output[1]);

    // Cleanup
    free(output);
}

void test_linear_basic2(void)
{
    float input[] = {1.0, 2.0, 3.0};
    float *weights[] = {(float[]){1.0, 2.0, 3.0}, (float[]){4.0, 5.0, 6.0}};
    float biases[] = {0.1, 0.2};
    float *output = linear(input, weights, biases, 3, 2);
    TEST_ASSERT_EQUAL_FLOAT(14.1, output[0]);
    TEST_ASSERT_EQUAL_FLOAT(32.2, output[1]);

    // Cleanup
    free(output);
}

// Add more test cases as needed
// implemented with ChatGPT help
void test_linear_zeros_and_negatives(void)
{
    float input[] = {0.0f, -1.0f, 2.0f};
    float *weights[] = {(float[]){0.0f, 0.0f, 0.0f}, (float[]){-1.0f, 1.0f, 0.5f}};
    float biases[] = {0.0f, 0.5f};
    float *output = linear(input, weights, biases, 3, 2);

    // first output is zero input times zero weights + 0 bias
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, output[0]);

    // second output = (0*-1) + (-1*1) + (2*0.5) + 0.5 = 0 + -1 + 1 + 0.5 = 0.5
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.5f, output[1]);

    free(output);
}

void test_linear_larger_matrix(void)
{
    int inSize = 4;
    int outSize = 3;
    float input[] = {1, 2, 3, 4};
    // allocate weight rows
    float *weights_mem = (float *)malloc(outSize * inSize * sizeof(float));
    float **weights = (float **)malloc(outSize * sizeof(float*));
    for (int i = 0; i < outSize; ++i) {
        weights[i] = &weights_mem[i * inSize];
        for (int j = 0; j < inSize; ++j) weights[i][j] = (float)(i + j + 1); // some pattern
    }
    float biases[] = {0.1f, 0.2f, 0.3f};
    float *output = linear(input, weights, biases, inSize, outSize);

    // basic sanity checks
    for (int i = 0; i < outSize; ++i) {
        TEST_ASSERT_NOT_NULL(output);
    }

    free(weights);
    free(weights_mem);
    free(output);
}