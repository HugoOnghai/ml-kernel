#include "conv.h"

// Im2col algorithm
float **im2col(float ***image, int numChannels, int imageSize, int kernelSize, int stride, int *outputSize)
{
    // ChatGPT was used to help me rubber duck my algorithm/logic
    // TODO: Implement the im2col algorithm
    if (numChannels > 1) { return NULL; }
    if (stride <= 0) { return NULL; }
    if (kernelSize > imageSize) { return NULL; }
    
    // as per Ed Discussion, assume the image only has one channel (numChannels == 1)

    int outDim = ((imageSize - kernelSize) / stride) + 1; // number of patches per row (or per column too since its square)
    int numPatches = outDim * outDim;
    int patchSize = kernelSize * kernelSize;

    // allocate space for the output
    float **output = (float **)malloc(patchSize * sizeof(float *));
    for (int i = 0; i < patchSize; i++)
    {
        output[i] = (float *)malloc(numPatches * sizeof(float));
    }

    // Im2col gets the output for one kernel at a time, so for multiple kernel convolutions, we must run this function multiple times.
    // two outermost for-loops drag kernel across the image
    for (int kernel_topleft_row = 0; kernel_topleft_row <= imageSize - kernelSize; kernel_topleft_row += stride)
    {
        for (int kernel_topleft_col = 0; kernel_topleft_col <= imageSize - kernelSize; kernel_topleft_col += stride)
        {
            // map the kernel position to a unique 1D range of output columns
            // each output column corresponds to a unique patch
            int curr_output_col = (kernel_topleft_row/stride) * outDim + (kernel_topleft_col/stride);

            // these two for-loops iterate through each pixel captured by the kernel at its current location
            for (int i = 0; i < kernelSize; i++)
            {
                for (int j = 0; j < kernelSize; j++)
                {
                    // map the current kernel pixel to a unique row in the output column
                    // each row j in an output column corresponds to the j-th value captured by that patch
                    int curr_output_row = i * kernelSize + j;

                    // the current kernel pixel w.r.t. the entire image's indices is
                    // topleft row and topleft col offset by the kernel pixel's relative indexing
                    output[curr_output_row][curr_output_col] = image[0][kernel_topleft_row + i][kernel_topleft_col + j];
                }
            }
        }
    }

    *outputSize = numPatches;
    return output; // output is size patchSize x numPatches = kernelSize*kernelSize x outDim*outDim
}

// Im2col algorithm's inverse
float ***col2im(float **result, int num_kernels, int conv_rows, int conv_cols)
{
    // TODO: Implement the col2im algorithm
    // Allocate memory for the image[num_kernels][conv_rows][conv_cols]
    float ***output = (float ***)malloc((size_t)num_kernels * sizeof(*output));
    for (int k = 0; k < num_kernels; k++)
    {
	    output[k] = (float **)malloc((size_t)conv_rows * sizeof(*output[k])); 
    	for (int r = 0; r < conv_rows; r++)
	    {
		    output[k][r] = (float *)calloc((size_t)(conv_cols), sizeof(*output[k][r]));
	    }
    }

    for (int k = 0; k < num_kernels; k++)
    {
        for (int p = 0; p < conv_rows * conv_cols; p++) // iterate through the columns which represent distinct patches
        {
            int output_row_idx = p / conv_cols; // where conv_cols is also the number of elements per row 
            int output_col_idx = p % conv_cols; // which gives us the offset from the 0th columns

            output[k][output_row_idx][output_col_idx] = result[k][p];
        }
    }

    return output;
}

float **kernel_flatten(float ****kernel, int num_kernels, int kernel_size)
{
    float **flattened_kernels = (float **)malloc(num_kernels * sizeof(float *));
    for (int i = 0; i < num_kernels; i++)
    {
        flattened_kernels[i] = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    }

    for (int k = 0; k < num_kernels; k++)
    {
        for (int i = 0; i < kernel_size; i++)
        {
            for (int j = 0; j < kernel_size; j++)
            {
                flattened_kernels[k][i * kernel_size + j] = kernel[k][0][i][j];
            }
        }
    }

    return flattened_kernels;
}

// Basic convolution operation
float ***convolution(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize)
{
    // TODO: Implement the convolution operation
    // allocate memory for the output image [numFilters][inputSize-kernelSize+1][inputSize-kernelSize+1]
    int outputSize = inputSize - kernelSize + 1;

    float ***output = (float ***)malloc((size_t)numFilters * sizeof(*output));
    for (int f = 0; f < numFilters; f++)
    {
	    output[f] = (float **)malloc((size_t)outputSize * sizeof(*output[f])); 
    	    for (int r = 0; r < outputSize; r++)
	    {
		    output[f][r] = (float *)calloc((size_t)(outputSize), sizeof(*output[f][r]));
	    }
    }
	
    for (int f = 0; f < numFilters; f++)
    {
	    for (int i = 0; i < outputSize; i++)
	    {
		    for (int j = 0; j < outputSize; j++)
		    {
			    float elem = 0;
			    for (int c = 0; c < numChannels; c++)
			    {
				    for (int ki = 0; ki < kernelSize; ki++)
				    {
					    for (int kj = 0; kj < kernelSize; kj++)
					    {
						    elem += image[c][i+ki][j+kj] * kernel[f][c][ki][kj];
					    }
				    }
			    }
			    output[f][i][j] = elem + biasData[f];
		    }
	    }
    }

    return output;
}

// Convolution with im2col algorithm
float ***convolution_im2col(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize, MatmulType matmul_type)
{
    // TODO: Implement the convolution operation with im2col algorithm
    // Flatten kernel
    float **flat_kernel = kernel_flatten(kernel, numFilters, kernelSize);

    // Apply im2col (ASSUMING STRIDE IS 1)
    int stride = 1;
    int numPatches = 0;
    float **flat_input = im2col(image, numChannels, inputSize, kernelSize, stride, &numPatches);

    // Apply matmul
    int patchSize = kernelSize * kernelSize; // number of patches in a given channel. numChannels is assumed 1.

    float **C;
    if (matmul_type == MATMUL_BASE)
    {
        C = matmul(flat_kernel, flat_input, numFilters, patchSize, patchSize, numPatches);
    }
    else if (matmul_type == MATMUL_SPARSE)
    {
        C = matmul_sparse(flat_kernel, flat_input, numFilters, patchSize, patchSize, numPatches);
    }

    // Apply col2im
    int outDim = ((inputSize - kernelSize) / stride) + 1; // sqrt(numPatches)
    if (numPatches != outDim * outDim) { return NULL; } // error checking...
    float ***output = col2im(C, numFilters, outDim, outDim);

    // Add bias and apply ReLU
    for (int filter = 0; filter < numFilters; filter++)
    {
        for (int conv_row = 0; conv_row < outDim; conv_row++)
        {
            for (int conv_col = 0; conv_col < outDim; conv_col++)
            {
                output[filter][conv_row][conv_col] += biasData[filter];
                output[filter][conv_row][conv_col] = relu(output[filter][conv_row][conv_col]);
            }
        }
    }

    // Cleanup
    // cleanup flat_kernel
    for (int k = 0; k < numFilters; k++)
    {
        free(flat_kernel[k]);
    }
    free(flat_kernel);

    // cleanup flat_input
    for (int i = 0; i < patchSize; i++) {
        free(flat_input[i]);
    }
    free(flat_input);
    
    // cleanup C
    for (int c = 0; c < numFilters; c++) {
        free(C[c]);
    }
    free(C);

    return output;
}