#include "conv.h"

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
