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

float **matmul_sparse(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    
}

float **csrmul(float **csr_A, float **csr_Bt, int A_rows, int A_cols, int B_rows, int B_cols)
{
    // allocate memory for the rows (each size of a float array)
    float **C = (float**)malloc((size_t)A_rows * sizeof(float*)); 

    // allocate memory for the cols of each row (each size of a float)
    for (int i = 0; i < A_rows; i++)
    {
	    C[i] = (float*)calloc((size_t)B_cols, sizeof(float));
    }

    for 
}
