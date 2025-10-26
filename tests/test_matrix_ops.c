#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "test_matrix_ops.h"
#include <stdio.h>

#define EPSILON 0.000001f

void assert_float_array_equal_matmul(float **expected, float **actual, int rows, int cols)
{
    TEST_ASSERT_NOT_NULL_MESSAGE(expected, "expected matrix is NULL");
    TEST_ASSERT_NOT_NULL_MESSAGE(actual,   "actual matrix is NULL");
    TEST_ASSERT_TRUE_MESSAGE(rows > 0 && cols > 0,  "rows/cols must be > 0");

    for (int i = 0; i < rows; i++)
    {
        /* If either of these fires, your row pointers are bogus */
        TEST_ASSERT_NOT_NULL_MESSAGE(expected[i], "expected[i] is NULL");
        TEST_ASSERT_NOT_NULL_MESSAGE(actual[i],   "actual[i] is NULL");

        for (int j = 0; j < cols; j++)
        {
            const float e = expected[i][j];   /* likely crash site */
            const float a = actual[i][j];

            char msg[128];
            (void)snprintf(msg, sizeof msg,
                           "Mismatch at (%d,%d): expected=%g actual=%g (eps=%g)",
                           i, j, (double)e, (double)a, (double)EPSILON);

            UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, e, a, __LINE__, msg);
        }
        UNITY_OUTPUT_CHAR('\n');
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

// =========================== MATMUL With BLOCKING ====================================
// =====================================================================================
void test_matmul_blocking_square_matrices(void)
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
    float **C = matmul_blocking(A, B, 2, 2, 2, 2);

    TEST_ASSERT_NOT_NULL_MESSAGE(C, "C is NULL");
    for (int i = 0; i < 2; ++i) 
    {
        TEST_ASSERT_NOT_NULL_MESSAGE(C[i], "C[i] is NULL (matmul returned bad rows)");
    }

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

void test_matmul_blocking_incompatible_dimensions(void)
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
    float **C = matmul_blocking(A, B, 2, 3, 2, 2);

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
void test_matmul_blocking_non_square_and_identity(void)
{
    // A is 2x3, B is 3x2 -> result 2x2
    float **A = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++) A[i] = (float *)malloc(3 * sizeof(float));
    A[0][0] = 1; A[0][1] = 1; A[0][2] = 1;
    A[1][0] = 1; A[1][1] = 1; A[1][2] = 1;

    float **B = (float **)malloc(3 * sizeof(float *));
    for (int i = 0; i < 3; i++) B[i] = (float *)malloc(2 * sizeof(float));
    B[0][0] = 1; B[0][1] = 1;
    B[1][0] = 1; B[1][1] = 1;
    B[2][0] = 1; B[2][1] = 1;

    float **C = matmul_blocking(A, B, 2, 3, 3, 2);
    TEST_ASSERT_NOT_NULL(C);
    // Expected C = [[3, 3],[3, 3]]
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, C[0][0], __LINE__, "C00 mismatch");
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, C[0][1], __LINE__, "C01 mismatch");
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, C[1][0], __LINE__, "C10 mismatch");
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, C[1][1], __LINE__, "C11 mismatch");

    // A is 2x3, B is 3x2 -> result 2x2
    A[0][0] = 1; A[0][1] = 2; A[0][2] = 3;
    A[1][0] = 4; A[1][1] = 5; A[1][2] = 6;

    B[0][0] = 7; B[0][1] = 8;
    B[1][0] = 9; B[1][1] = 10;
    B[2][0] = 11; B[2][1] = 12;

    C = matmul_blocking(A, B, 2, 3, 3, 2);
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
    float **R = matmul_blocking(I, M, 2, 2, 2, 2);
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, R[0][0], __LINE__, "R00");
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, R[0][1], __LINE__, "R01");
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.0f, R[1][0], __LINE__, "R10");
    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6.0f, R[1][1], __LINE__, "R11");

    // Cleanup multiple allocations
    for (int i = 0; i < 2; i++) { free(A[i]); free(I[i]); free(M[i]); free(C[i]); free(R[i]); }
    for (int i = 0; i < 3; i++) free(B[i]);
    free(A); free(B); free(C); free(I); free(M); free(R);
}

// =========================== MATMUL With QUANTIZATION ====================================
// =====================================================================================
void test_extract_be(void)
{
    // Setup
    unsigned int **A = (unsigned int **)malloc(3 * sizeof(unsigned int *));
    for (int i = 0; i < 3; i++)
    {
        A[i] = (unsigned int *)malloc(sizeof(unsigned int));
    }
    A[0][0] = 0x01020304;
    A[1][0] = 0x05060708;
    A[2][0] = 0x090A0B0C;

    TEST_ASSERT_EQUAL_UINT8(extract_be(A[0][0], 0), 1);
    TEST_ASSERT_EQUAL_UINT8(extract_be(A[0][0], 1), 2);
    TEST_ASSERT_EQUAL_UINT8(extract_be(A[0][0], 2), 3);
    TEST_ASSERT_EQUAL_UINT8(extract_be(A[0][0], 3), 4);

    TEST_ASSERT_EQUAL_UINT8(extract_be(A[1][0], 0), 5);
    TEST_ASSERT_EQUAL_UINT8(extract_be(A[1][0], 1), 6);
    TEST_ASSERT_EQUAL_UINT8(extract_be(A[1][0], 2), 7);
    TEST_ASSERT_EQUAL_UINT8(extract_be(A[1][0], 3), 8);

    TEST_ASSERT_EQUAL_UINT8(extract_be(A[2][0], 0), 9);
    TEST_ASSERT_EQUAL_UINT8(extract_be(A[2][0], 1), 10);
    TEST_ASSERT_EQUAL_UINT8(extract_be(A[2][0], 2), 11);
    TEST_ASSERT_EQUAL_UINT8(extract_be(A[2][0], 3), 12);
}

void test_matmul_quant_smallest_matrices(void)
{
    unsigned int **A = (unsigned int **)malloc(1 * sizeof(unsigned int *));
    for (int i = 0; i < 1; i++)
    {
        A[i] = (unsigned int *)malloc(sizeof(unsigned int));
    }
    
    A[0][0] = 0xFFFFFFFF;

    int **B = (int **)malloc(4 * sizeof(int *));
    for (int i = 0; i < 4; i++)
    {
        B[i] = (int *)malloc(sizeof(int)); 
    }
     
    B[0][0] = 1;
    B[1][0] = 1;
    B[2][0] = 1;
    B[3][0] = 1;

    float** C = matmul_quant(A, B, 1, 1, 4, 1);

    UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1020, C[0][0], __LINE__, "C00 mismatch");
}

void test_matmul_quant_example_matrices(void)
{
    // Setup
    unsigned int **A = (unsigned int **)malloc(3 * sizeof(unsigned int *));
    for (int i = 0; i < 3; i++)
    {
        A[i] = (unsigned int *)malloc(sizeof(unsigned int));
    }
    A[0][0] = 0x01010101;
    A[1][0] = 0x01010101;
    A[2][0] = 0x01010101;

    int **B = (int **)malloc(4 * sizeof(int *));
    for (int i = 0; i < 4; i++)
    {
        B[i] = (int *)malloc(2 * sizeof(int));
    }
    B[0][0] = 2;
    B[0][1] = 0;
    B[1][0] = 1;
    B[1][1] = 2;
    B[2][0] = 1;
    B[2][1] = 2;
    B[3][0] = 1;
    B[3][1] = 2;

    float **expected = (float **)malloc(3 * sizeof(float *));
    for (int i = 0; i < 3; i++)
    {
        expected[i] = (float *)malloc(2 * sizeof(float));
    }
    expected[0][0] = 5.0f;
    expected[0][1] = 6.0f;
    expected[1][0] = 5.0f;
    expected[1][1] = 6.0f;
    expected[2][0] = 5.0f;
    expected[2][1] = 6.0f;

    TEST_ASSERT_NOT_NULL_MESSAGE(A, "expected matrix is NULL");
    TEST_ASSERT_NOT_NULL_MESSAGE(B, "actual matrix is NULL");
    TEST_ASSERT_NOT_NULL_MESSAGE(expected, "actual matrix is NULL");

    // Run function under test
    float **C = matmul_quant(A, B, 3, 1, 4, 2);

    // Check expectations
    assert_float_array_equal_matmul(expected, C, 3, 2);

    // // Cleanup
    // for (int i = 0; i < 3; i++)
    // {
    //     free(A[i]);
    // }
    // for (int i = 0; i < 4; i++)
    // {
    //     free(B[i]);
    // }
    // for (int i = 0; i < 32; i++)
    // {
    //     free(C[i]);
    // }
    // for (int i = 0; i < 3; i++)
    // {
    //     free(expected[i]);
    // }   
    // free(A);
    // free(B);
    // free(C);
    // free(expected);
}

void test_matmul_quant_incompatible_dimensions(void)
{
    // Setup
    unsigned int **A = (unsigned int **)malloc(2 * sizeof(unsigned int *));
    for (int i = 0; i < 2; i++)
    {
        A[i] = (unsigned int *)malloc(3 * sizeof(unsigned int));
    }

    int **B = (int **)malloc(2 * sizeof(int *));
    for (int i = 0; i < 2; i++)
    {
        B[i] = (int *)malloc(2 * sizeof(int));
    }

    // Run function under test
    float **C = matmul_quant(A, B, 2, 3, 2, 2);

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

void test_matmul_quant_non_square(void)
{
    // let's test a A = 2 x 2 and B = 8 x 3
    int A_rows = 2;
    int A_cols = 2;
    int B_rows = 8;
    int B_cols = 3;

    unsigned int ** A_quant = (unsigned int **)malloc(A_rows * sizeof(unsigned int *));
    for (int i = 0; i < A_rows; i++)
    {
        A_quant[i] = (unsigned int *)malloc(A_cols * sizeof(unsigned int));
    }

    A_quant[0][0] = 0x01010101; A_quant[0][1] = 0x01010101;
    A_quant[1][0] = 0x01010101; A_quant[1][1] = 0x01010101;

    float ** A = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++)
    {
        A[i] = (float *)malloc(A_cols * 4 * sizeof(float));
    }
    
    A[0][0] = 1.0f; A[0][1] = 1.0f; A[0][2] = 1.0f; A[0][3] = 1.0f; A[0][4] = 1.0f; A[0][5] = 1.0f; A[0][6] = 1.0f; A[0][7] = 1.0f;
    A[1][0] = 1.0f; A[1][1] = 1.0f; A[1][2] = 1.0f; A[1][3] = 1.0f; A[1][4] = 1.0f; A[1][5] = 1.0f; A[1][6] = 1.0f; A[1][7] = 1.0f;

    int ** B_int = (int **)malloc(B_rows * sizeof(int *));
    for (int i = 0; i < B_rows; i++)
    {
        B_int[i] = (int *)malloc(B_cols * sizeof(int));
    }

    for (int i = 0; i < B_rows; i++)
    {
        for (int j = 0; j < B_cols; j++)
        {
            B_int[i][j] = 1;
        }
    }

    float ** B = (float **)malloc(B_rows * sizeof(float *));
    for (int i = 0; i < B_rows; i++)
    {
        B[i] = (float *)malloc(B_cols * sizeof(float));
    }

    for (int i = 0; i < B_rows; i++)
    {
        for (int j = 0; j < B_cols; j++)
        {
            B[i][j] = 1.0f;
        }
    }

    float** C_from_quant = matmul_quant(A_quant, B_int, A_rows, A_cols, B_rows, B_cols);
    float** C_from_naive = matmul(A, B, A_rows, 4*A_cols, B_rows, B_cols);

    assert_float_array_equal_matmul(C_from_quant, C_from_naive, A_rows, B_cols);
}