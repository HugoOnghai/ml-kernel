#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

float **matmul(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
float **matmul_blocking(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
float **matmul_quant(unsigned int **A, int **B, int A_rows, int A_cols, int B_rows, int B_cols);
void dense2csr(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, 
                float **nonzeros_A, int **rowptr_A, int **colidx_A, 
                float **nonzeros_B, int **rowptr_B, int **colidx_B);
float **matmul_sparse(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
float **csrmul(float *nonzeros_A, int *rowptr_A, int *colidx_A, 
               float *nonzeros_B, int *rowptr_B, int *colidx_B,
               int A_rows, int A_cols, int B_rows, int B_cols);
uint8_t extract_be(uint32_t a, int idx);

#endif /* MATRIX_OPS_H */