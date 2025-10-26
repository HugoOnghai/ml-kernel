#ifndef TEST_MATRIX_OPS_H
#define TEST_MATRIX_OPS_H

void test_matmul_square_matrices(void);
void test_matmul_incompatible_dimensions(void);
void test_matmul_non_square_and_identity(void);

void test_matmul_blocking_square_matrices(void);
void test_matmul_blocking_incompatible_dimensions(void);
void test_matmul_blocking_non_square_and_identity(void);

void test_matmul_quant_example_matrices(void);
void test_matmul_quant_incompatible_dimensions(void);
void test_matmul_quant_non_square(void);
void test_matmul_quant_smallest_matrices(void);
void test_extract_be(void);

#endif /* TEST_MATRIX_OPS_H */
