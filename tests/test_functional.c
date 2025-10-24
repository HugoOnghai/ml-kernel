#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "test_functional.h"
#include <float.h>


void test_softmax_basic(void) {
    float input[] = {1.0, 2.0, 3.0};
    float *output = softmax(input, 3);
    float sum = 0.0;

    // Check that the sum of the output is 0 (because the output is log softmax)
    for (int i = 0; i < 3; i++) {
        sum += expf(output[i]);
    }

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0, sum);

    // Check that the maximum input corresponds to the maximum output
    int maxInputIndex = 0;
    int maxOutputIndex = 0;

    for (int i = 1; i < 3; i++) {
        if (input[i] > input[maxInputIndex]) {
            maxInputIndex = i;
        }
        if (output[i] > output[maxOutputIndex]) {
            maxOutputIndex = i;
        }
    }

    TEST_ASSERT_EQUAL_INT(maxInputIndex, maxOutputIndex);

    // Cleanup
    free(output);
}

void test_relu(void) {
    float inputs[] = {3.0f, 0.0f, -3.0f};
    float expected_outputs[] = {3.0f, 0.0f, 0.0f};
    int test_cases = sizeof(inputs)/sizeof(inputs[0]);

    for(int i = 0; i < test_cases; i++) {
        float output = relu(inputs[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_outputs[i], output);
    }
}


// Add more test cases as needed

void test_softmax_large_inputs(void) {
    // Ensure numerical stability for large inputs (log-softmax expected)
    float input[] = {1000.0f, 1001.0f, 1002.0f};
    float *output = softmax(input, 3);
    float sum = 0.0f;

    for (int i = 0; i < 3; i++) {
        sum += expf(output[i]);
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, sum);

    // The largest input should correspond to the largest output
    int maxInputIndex = 0;
    int maxOutputIndex = 0;
    for (int i = 1; i < 3; i++) {
        if (input[i] > input[maxInputIndex]) maxInputIndex = i;
        if (output[i] > output[maxOutputIndex]) maxOutputIndex = i;
    }
    TEST_ASSERT_EQUAL_INT(maxInputIndex, maxOutputIndex);

    free(output);
}

void test_applyRelu_array(void) {
    float arr[] = {-1.5f, 0.0f, 2.5f};
    applyRelu(arr, 3);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, arr[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, arr[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 2.5f, arr[2]);
}

// test that softmax outputs sum to 1
void test_normalized_softmax(void) {
    float input[] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, -3.0f};
    int inputSize = sizeof(input)/sizeof(input[0]);
    float *output = softmax(input, inputSize);
    float sum = 0.0f;
    for (int i = 0; i < inputSize; ++i) {
        sum += expf(output[i]);
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, sum);
    free(output);
}