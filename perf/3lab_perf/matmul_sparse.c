#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../../kernel/matrix_ops.h"

float **generate_random_matrix_with_sparsity(int rows, int cols, float sparsity) {
    if (sparsity > 1 || sparsity < 0) { return NULL; }

    float **matrix = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float *)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++) {

            float threshold = (float)RAND_MAX * sparsity;
            // "bernoulli distribution with probability sparsity of making this entry sparse"
            if (rand() > threshold) // if the random number is above the threshold, it won't be sparse. 
            {
                matrix[i][j] = (float)rand() / RAND_MAX;
            }
            else // odd
            {
                matrix[i][j] = 0.0f;
            }
        }
    }
    return matrix;
}

void free_matrix(float **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main() {
    srand(time(NULL));
    int A_rows = 28, A_cols = 28, B_rows = 28, B_cols = 28;
    // int A_rows = 56, A_cols = 56, B_rows = 56, B_cols = 56;
    // int A_rows = 224, A_cols = 224, B_rows = 224, B_cols = 224;
    // int A_rows = 1120, A_cols = 1120, B_rows = 1120, B_cols = 1120;
    float sparsity = 0.1;

    if (A_cols != B_rows) {
        printf("Matrix dimensions incompatible for multiplication.\n");
        return 1;
    }

    float **A = generate_random_matrix_with_sparsity(A_rows, A_cols, sparsity);
    float **B = generate_random_matrix_with_sparsity(B_rows, B_cols, sparsity);

    printf("Converting my matrices into their CSR forms!...\n");

    float *nonzeros_A;
    int *rowptr_A;
    int *colidx_A;

    float *nonzeros_B;
    int *rowptr_B;
    int *colidx_B;

    dense2csr(A, B, A_rows, A_cols, B_rows, B_cols, &nonzeros_A, &rowptr_A, &colidx_A, &nonzeros_B, &rowptr_B, &colidx_B);

    printf("Compute csrmul 5 times...");

    float **C;

    for (int i = 0; i < 5; i++)
    {
        C = csrmul(nonzeros_A, rowptr_A, colidx_A, nonzeros_B, rowptr_B, colidx_B, A_rows, A_cols, B_rows, B_cols);
    }

    // Cleanup
    free_matrix(A, A_rows);
    free_matrix(B, B_rows);
    free_matrix(C, A_rows);

    free(nonzeros_A);
    free(rowptr_A);
    free(colidx_A);
    free(nonzeros_B);
    free(rowptr_B);
    free(colidx_B);

    printf("matmul_sparse completed.\n");

    return 0;
}
