#include "unity/unity.h"
#include "test_conv.h"
// #include "test_nn.h"
#include "test_functional.h"
#include "test_linear.h"
#include "test_matrix_ops.h"
#include "test_attention.h"

void setUp(void) {
    /* Code here will run before each test */
}

void tearDown(void) {
    /* Code here will run after each test */
}

int main(void) {
    UNITY_BEGIN();

    // // Test conv
    RUN_TEST(test_conv);
    RUN_TEST(test_conv_multiple_filters_and_bias);
    RUN_TEST(test_convolution_im2col);
    RUN_TEST(test_convolution_im2col_multiple_filters_and_bias);

    // Test nn
    // RUN_TEST(test_flatten_basic);
    // RUN_TEST(test_predict_simple_array);
    // RUN_TEST(test_predict_all_same_values);
    // RUN_TEST(test_predict_mix_of_negatives_and_positives);

    // Test functional
    RUN_TEST(test_softmax_basic);
    RUN_TEST(test_normalized_softmax);
    RUN_TEST(test_relu);
    RUN_TEST(test_softmax_large_inputs);
    RUN_TEST(test_applyRelu_array);

    // Test linear
    RUN_TEST(test_linear_basic);
    RUN_TEST(test_linear_basic2);
    RUN_TEST(test_linear_zeros_and_negatives);
    RUN_TEST(test_linear_larger_matrix);

    // Test matrix_ops
    RUN_TEST(test_matmul_square_matrices);
    RUN_TEST(test_matmul_incompatible_dimensions);
    RUN_TEST(test_matmul_non_square_and_identity);

    // Test matrix_ops with blocking
    RUN_TEST(test_matmul_blocking_square_matrices);
    RUN_TEST(test_matmul_blocking_incompatible_dimensions);
    RUN_TEST(test_matmul_blocking_non_square_and_identity);

    // // Test matrix_ops with quantization
    RUN_TEST(test_extract_be);
    RUN_TEST(test_matmul_quant_example_matrices);
    RUN_TEST(test_matmul_quant_smallest_matrices);
    RUN_TEST(test_matmul_quant_incompatible_dimensions);
    RUN_TEST(test_matmul_quant_non_square);

    // Test matrix_ops with sparse formulation
    RUN_TEST(test_matmul_sparse_square_matrices);
    RUN_TEST(test_matmul_sparse_incompatible_dimensions);
    RUN_TEST(test_matmul_sparse_non_square_and_identity);

    // Test attention
    RUN_TEST(test_scaled_dot_product_attention);

    return UNITY_END();
}
