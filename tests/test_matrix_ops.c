#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "test_matrix_ops.h"


#define EPSILON 0.000001f

void assert_float_array_equal_matmul(float **expected, float **actual, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected[i][j], actual[i][j], __LINE__, "Arrays Not Equal!");
        }
    }
}

void test_matmul_square_matrices(void)
{
    // Setup
    float **A = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        A[i] = (float *)malloc(2 * sizeof(float));
    }
    A[0][0] = 1.0f;
    A[0][1] = 2.0f;
    A[1][0] = 3.0f;
    A[1][1] = 4.0f;

    float **B = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        B[i] = (float *)malloc(2 * sizeof(float));
    }
    B[0][0] = 2.0f;
    B[0][1] = 0.0f;
    B[1][0] = 1.0f;
    B[1][1] = 2.0f;

    float **expected = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        expected[i] = (float *)malloc(2 * sizeof(float));
    }
    expected[0][0] = 4.0f;
    expected[0][1] = 4.0f;
    expected[1][0] = 10.0f;
    expected[1][1] = 8.0f;

    // Run function under test
    float **C = matmul(A, B, 2, 2, 2, 2);

    // Check expectations
    assert_float_array_equal_matmul(expected, C, 2, 2);

    // Cleanup
    for (int i = 0; i < 2; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
        free(expected[i]);
    }
    free(A);
    free(B);
    free(C);
    free(expected);
}

void test_matmul_incompatible_dimensions(void)
{
    // Setup
    float **A = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        A[i] = (float *)malloc(3 * sizeof(float));
    }

    float **B = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        B[i] = (float *)malloc(2 * sizeof(float));
    }

    // Run function under test
    float **C = matmul(A, B, 2, 3, 2, 2);

    // Check expectations
    UNITY_TEST_ASSERT_NULL(C, __LINE__, "Expected NULL!");

    // Cleanup
    for (int i = 0; i < 2; i++)
    {
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);
}

// implemented with ChatGPT
void test_matmul_non_square_and_identity(void)
{
    // A is 2x3, B is 3x2 -> result 2x2
    float **A = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++) A[i] = (float *)malloc(3 * sizeof(float));
    A[0][0] = 1; A[0][1] = 2; A[0][2] = 3;
    A[1][0] = 4; A[1][1] = 5; A[1][2] = 6;

    float **B = (float **)malloc(3 * sizeof(float *));
    for (int i = 0; i < 3; i++) B[i] = (float *)malloc(2 * sizeof(float));
    B[0][0] = 7; B[0][1] = 8;
    B[1][0] = 9; B[1][1] = 10;
    B[2][0] = 11; B[2][1] = 12;

    float **C = matmul(A, B, 2, 3, 3, 2);
    TEST_ASSERT_NOT_NULL(C);
    // Expected C = [[58, 64], [139, 154]]
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 58.0f, C[0][0], __LINE__, "C00 mismatch");
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 64.0f, C[0][1], __LINE__, "C01 mismatch");
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 139.0f, C[1][0], __LINE__, "C10 mismatch");
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 154.0f, C[1][1], __LINE__, "C11 mismatch");

    // Identity test: multiply 2x2 identity with a 2x2 matrix
    float **I = (float **)malloc(2 * sizeof(float *));
    float **M = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++) { I[i] = (float *)malloc(2 * sizeof(float)); M[i] = (float *)malloc(2 * sizeof(float)); }
    I[0][0]=1; I[0][1]=0; I[1][0]=0; I[1][1]=1;
    M[0][0]=3; M[0][1]=4; M[1][0]=5; M[1][1]=6;
    float **R = matmul(I, M, 2, 2, 2, 2);
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, R[0][0], __LINE__, "R00");
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, R[0][1], __LINE__, "R01");
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.0f, R[1][0], __LINE__, "R10");
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6.0f, R[1][1], __LINE__, "R11");

    // Cleanup multiple allocations
    for (int i = 0; i < 2; i++) { free(A[i]); free(I[i]); free(M[i]); free(C[i]); free(R[i]); }
    for (int i = 0; i < 3; i++) free(B[i]);
    free(A); free(B); free(C); free(I); free(M); free(R);
}