#include "matrix_ops.h"

uint8_t extract_be(uint32_t a, int idx) 
{
    // big-endian logical order
    return (uint8_t)((a >> ((3 - idx) * 8)) & 0xFF);
}

float **matmul(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols) 
{
    // TODO: Implement matrix multiplication
    if (A_cols != B_rows) { return NULL; }

    // allocate memory for the rows (each size of a float array)
    float **C = (float**)malloc((size_t)A_rows * sizeof(float*)); 

    // allocate memory for the cols of each row (each size of a float)
    for (int i = 0; i < A_rows; i++)
    {
	    C[i] = (float*)calloc((size_t)B_cols, sizeof(float));
    }

    for (int i = 0; i < A_rows; i++)
    {
	    for (int j = 0; j < B_cols; j++)
	    {
		    for (int k = 0; k < A_cols; k++)
		    {
			    C[i][j] += A[i][k] * B[k][j];
		    }
	    }
    }
    return C;
}

#define BLOCK 224

float **matmul_blocking(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    // sources:
    // https://iitd.github.io/col729/lec/matrix_multiplication.html
    // https://www.reddit.com/r/learnprogramming/comments/le4ve/how_does_blocking_increase_speedup_of_matrix/

    if (A_cols != B_rows) { return NULL; }

    // allocate memory for the rows (each size of a float array)
    float **C = (float**)malloc((size_t)A_rows * sizeof(float*)); 

    // allocate memory for the cols of each row (each size of a float)
    for (int i = 0; i < A_rows; i++)
    {
	    C[i] = (float*)calloc((size_t)B_cols, sizeof(float)); // calloc initializes to zero.
    }

    int bi=0;
    int bj=0;
    int bk=0;

    int i=0;
    int j=0;
    int k=0;

    int blockSize = BLOCK;
	
    // implementation based on reddit post cited above.
    // basically blocking should be faster than naive because we are improving data locality (otherwise it's essentially the same)
    // by blocking, we only access one local block of a matrix at a time, so we have better cache hit rates
    for(bi = 0; bi<A_rows; bi+=blockSize)
    {
        // calculate i_max as soon as possible to avoid wasteful re-calculating
        int i_max = bi+blockSize;
        if (i_max > A_rows) { i_max = A_rows; }

	    for(bj = 0; bj<B_cols; bj+=blockSize)
		{   
            int j_max = bj+blockSize;
            if (j_max > B_cols) { j_max = B_cols; }

            for(bk = 0; bk<A_cols; bk+=blockSize)
            {
                int k_max = bk+blockSize;
                if (k_max > A_cols) { k_max = A_cols; }

                for(i = bi; i < i_max; i++)
                {
                    for(j = bj; j < j_max; j++)
                    {
                        for(k = bk; k < k_max; k++)
                        {
                            C[i][j] +=  A[i][k] * B[k][j];
                        }
                    }
                }

            }
        }
    }
    return C;
}

float **matmul_quant(unsigned int **A, int **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    if (A_rows <= 0 || A_cols <= 0 || B_rows <= 0 || B_cols <= 0) { return NULL;}
    if (4 * A_cols != B_rows) { return NULL; }

    // allocate memory for the rows (each size of a float array)
    float **C = (float**)malloc((size_t)A_rows * sizeof(float*)); 

    // allocate memory for the cols of each row (each size of a float)
    for (int i = 0; i < A_rows; i++)
    {
	    C[i] = (float*)calloc((size_t)B_cols, sizeof(float));
    }

    for (int i = 0; i < A_rows; i++)
    {
	    for (int j = 0; j < B_cols; j++)
	    {
		    for (int k = 0; k < B_rows; k++)
		    {
			    C[i][j] += (float)(extract_be(A[i][k/4], (k%4)) * B[k][j]);
		    }
	    }
    }
    return C;
}

// MATMUL SPARSE!!!!!!!!!!!!
void dense2csr(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, 
                float **nonzeros_A, int **rowptr_A, int **colidx_A, 
                float **nonzeros_B, int **rowptr_B, int **colidx_B)
{
    int num_nonzeros_A = 0;
    int num_nonzeros_B = 0;

    *rowptr_A = (int*)calloc((size_t)(A_rows+1), sizeof(int)); // always of size A_rows + 1 and always rowptr_A[0] == 0
    *rowptr_B = (int*)calloc((size_t)(B_rows+1), sizeof(int)); // always of size B_rows + 1 and always rowptr_B[0] == 0

    for (int i = 0; i < A_rows; i++)
    {
        for (int j = 0; j < A_cols; j++)
        {
            if (A[i][j] != 0.0f)
            {
                num_nonzeros_A++;
            }
        }
       (*rowptr_A)[i+1] = num_nonzeros_A; // after counting all the new nonzeros found after checking all entires of i-th row, update rowptr_A[i+1]
    }

    for (int i = 0; i < B_rows; i++)
    {
        for (int j = 0; j < B_cols; j++)
        {
            if (B[i][j] != 0.0f)
            {
                num_nonzeros_B++;
            }
        }
        (*rowptr_B)[i+1] = num_nonzeros_B; // after counting all the new nonzeros found after checking all entires of i-th row, update rowptr_B[i+1]
    }

    // now that we know the number of nonzero values, we can allocate space for two arrays of size num_nonzeros, for each matrix.
    *nonzeros_A = (float*)calloc((size_t)(num_nonzeros_A), sizeof(float));
    *nonzeros_B = (float*)calloc((size_t)(num_nonzeros_B), sizeof(float));

    *colidx_A = (int*)calloc((size_t)(num_nonzeros_A), sizeof(int));
    *colidx_B = (int*)calloc((size_t)(num_nonzeros_B), sizeof(int));

    int index_A = 0;
    for (int i = 0; i < A_rows; i++)
    {
        for (int j = 0; j < A_cols; j++)
        {
            if (A[i][j] != 0)
            {
                (*nonzeros_A)[index_A] = A[i][j];
                (*colidx_A)[index_A] = j;
                index_A++; // increment to next entry in arrays for next nonzero value.
            }
        }
    }

    int index_B = 0;
    for (int i = 0; i < B_rows; i++)
    {
        for (int j = 0; j < B_cols; j++)
        {
            if (B[i][j] != 0)
            {
                (*nonzeros_B)[index_B] = B[i][j];
                (*colidx_B)[index_B] = j;
                index_B++; // increment to next entry in arrays for next nonzero value.
            }
        }
    }
}

float **matmul_sparse(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    // int A_rows, int A_cols, int B_rows, int B_cols, 
                // float **nonzeros_A, int **rowptr_A, int **colidx_A, 
                // float **nonzeros_B, int **rowptr_B, int **colidx_B
    float *nonzeros_A;
    int *rowptr_A;
    int *colidx_A;

    float *nonzeros_B;
    int *rowptr_B;
    int *colidx_B;

    dense2csr(A, B, A_rows, A_cols, B_rows, B_cols, &nonzeros_A, &rowptr_A, &colidx_A, &nonzeros_B, &rowptr_B, &colidx_B);

    // now we have the three-array CSR form of A and B
    float **C = csrmul(nonzeros_A, rowptr_A, colidx_A, nonzeros_B, rowptr_B, colidx_B, A_rows, A_cols, B_rows, B_cols);

    free(nonzeros_A);
    free(rowptr_A);
    free(colidx_A);
    free(nonzeros_B);
    free(rowptr_B);
    free(colidx_B);

    return C;
}

float **csrmul(float *nonzeros_A, int *rowptr_A, int *colidx_A, 
               float *nonzeros_B, int *rowptr_B, int *colidx_B,
               int A_rows, int A_cols, int B_rows, int B_cols)
{
    if (A_cols != B_rows)
    {
        return NULL; // check the dimensions of the matrices we wish to multiply
    }

    // allocate memory for the rows (each size of a float array)
    float **C = (float**)malloc((size_t)A_rows * sizeof(float*)); 

    // allocate memory for the cols of each row (each size of a float)
    for (int i = 0; i < A_rows; i++)
    {
	    C[i] = (float*)calloc((size_t)B_cols, sizeof(float)); // calloc is malloc and then initialization to zero
    }

    int rowpos_C = 0;
    int colpos_C = 0;

    // iterate over the rows of A
    for (int i = 0; i < A_rows; i++)
    {
        // iterate through the nonzero values in the i-th row of A
        // p represents the index of that nonzero value in nonzeros_A
        for (int p = rowptr_A[i]; p < rowptr_A[i + 1]; p++)
        { 
            int k = colidx_A[p]; // the columnar index of each nonzero value
            float A_ik = nonzeros_A[p]; // the nonzero value at the i-th row and k-th column of A

            // instead of focusing on one element of C at a time, we instead do all multiplications of A_ik
            // that is, add A_ik * B_k1 to C[i][1], A_ik * B_k2 to C[i][2], ... A_ik * B_k2 to C[i][j]
            // this is easier because nonzeros_B csr(B) is grouped by rows, with index ranges defined in rowptr_B[k] to rowptr_B[k+1]
            for (int q = rowptr_B[k]; q < rowptr_B[k+1]; q++)
            {
                int j = colidx_B[q];
                float B_kj = nonzeros_B[q];

                C[i][j] += A_ik * B_kj;
            }
        }
    }
    return C;
}
