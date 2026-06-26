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

// Implemented in lab 4
// pass threaded function arguments via an argument struct
// declaring it above worker_func so it knows what it is
struct arg_struct {
    float **A;
    float **B;
    float *C_flat;
    int A_rows;
    int A_cols;
    int B_cols;
    int start_idx;
    int end_idx;
};

void *worker_func(void *arg) {
    // cast arg to our struct
    struct arg_struct *args = (struct arg_struct*)arg;

    // retrieve vals from struct
    float **A = args->A;
    float **B = args->B;
    float *C_flat = args->C_flat;

    int A_rows = args->A_rows;
    int A_cols = args->A_cols;
    int B_cols = args->B_cols;

    int start = args->start_idx;
    int end = args->end_idx;

    // Loop over assigned output elements (C_flat indexing)
    for (int idx = start; idx < end; idx++) {

        // idx is row major, so cols is remainder of num_cols, and every set of B_cols iterated through is a new row.
        int row = idx / B_cols;
        int col = idx % B_cols;

        float sum = 0.0f;

        // Naive matmul, same inner product for C[row][col]
        for (int k = 0; k < A_cols; k++) {
            sum += A[row][k] * B[k][col];
        }

        // write sum to C_flat (at given index)
        C_flat[idx] = sum;
    }
    return NULL;
}

float **matmul_thread(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    // TODO: Implement multithreaded matrix multiplication
    // notes on pthred_create() (from geeksforgeeks.org)
    // pthread_create(thread, attr, routine, arg);
    // thread = pointer to pthread_t variable that stores id of the thread
    // attr = thread attribute obj that defines thread properties
    // routine = pointer to func that thread will execute void* and accept a void* argument
    // arg = a single argument passed to the thread function (pass a struct or pointer to pass multiple vals)
    
    // pthread_t thread1;

    // pthread_join(thread, retval);
    // thread = thread Id you want to wait for (the thread that will be joined)
    // retval = pointer to location where the exit status of the thread will be stored.
    // synchronize threads to ensure execution of threads ends before main() ends, otherwise unexpected behavior

    // if dimensions mismatch, NULL
    if (A_cols != B_rows) { return NULL; }

    // allocate memory for the 2D matrix C
    float **C = (float**)malloc(A_rows * sizeof(float*));

    // allocate memory for the flattened C matrix, which we can then align to the start of a cache line.
    float *C_flat;

    // from man7.org, this allocates sizeof(float)*A_rows*B_cols bytes, starting with the memory address &C_flat.
    // and this starting memory address will be a multiple of 64 --> will be at the start of a cache line.
    // cache line is 64 bytes, which we found from "getconf LEVEL1_DCACHE_LINESIZE" (returned 64).
    posix_memalign((void**)&C_flat, 64, sizeof(float) * A_rows * B_cols); 
    memset(C_flat, 0, sizeof(float) * A_rows * B_cols); // this fills all of the values in the output matrix with 0, from <string.h>

    // from the lab guidelines, we know slurm allocates us 8 cores. Since each CPU core only can run 1 thread at a time,
    // let us initialize and distribute our task across all 8 threads, if we made extra threads (it'd just be wasted memory overhead, bad for perf)
    // Asked TA, should we use all cores even if it leads to false sharing (in the case of small input matrices)
    int num_threads = 8; // 1 thread per core
    int total_elems = A_rows * B_cols; // size of output matrix
    int num_cache_lines = (total_elems + 15) / 16; // add 15 to force truncation to "round up" to count the potentially last partially filled cache line.
    int min_cache_lines_per_thread = num_cache_lines / num_threads; // int division will cast to int, so handle "lost/remaining" lines next
    int total_remaining_cache_lines = num_cache_lines % num_threads; // add one remaining cache_line to one thread each, should be >=0 & <=7.

    pthread_t threads[num_threads];

    struct arg_struct args[num_threads];

    int curr_elem = 0;
    for (int t = 0; t < num_threads; ++t) {
        int num_cache_lines_per_thread = min_cache_lines_per_thread + (t < total_remaining_cache_lines ? 1 : 0); // gives a remaining cache line to the first "total_rem_cache_lines" of threads
        int num_elems_per_thread = num_cache_lines_per_thread * 16; // since cache_line_size/sizeof(float)=16

        int start_idx = curr_elem;
        int end_idx = start_idx + num_elems_per_thread;
        if (end_idx > total_elems) { end_idx = total_elems; } // handle case of partially filled last cache line
        curr_elem = end_idx; // prepare for next thread

        args[t] = (struct arg_struct){
            .A = A,
            .B = B,
            .C_flat = C_flat,
            .A_rows = A_rows,
            .A_cols = A_cols,
            .B_cols = B_cols,
            .start_idx = start_idx,
            .end_idx = end_idx
        };

        pthread_create(&threads[t], NULL, worker_func, &args[t]);
    }

    for (int t = 0; t < num_threads; ++t)
        pthread_join(threads[t], NULL);

    // Convert flat C to 2D C
    for (int i = 0; i < A_rows; ++i)
    {
        C[i] = (float*)malloc((size_t)B_cols * sizeof(float));
        for (int j = 0; j < B_cols; ++j) 
        {
            C[i][j] = C_flat[i * B_cols + j];
        }
    }

    free(C_flat); // clean up after writing to C 2D matrix

    return C;
}