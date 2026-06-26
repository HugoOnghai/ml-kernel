#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../kernel/conv.h"

int main() {
	int scale = 50;
	// use the functions built in test_conv
	
	// make arrays for my kernel, image, and output
	
	// kernel (allocate as float **** to match convolution signature)
	int numFilters = 3;
	int kernel_size = 3*scale;
	int numChannels = 3;

	float ****kernel = (float ****)malloc((size_t)numFilters * sizeof(float***));
	for (int f = 0; f < numFilters; f++){
		kernel[f] = (float ***)malloc((size_t)numChannels * sizeof(float**));
		for (int c = 0; c < numChannels; c++){
			kernel[f][c] = (float **)malloc((size_t)kernel_size * sizeof(float*));
			for (int i = 0; i < kernel_size; i++){
				kernel[f][c][i] = (float *)malloc((size_t)kernel_size * sizeof(float));
				for (int j = 0; j < kernel_size; j++){
					kernel[f][c][i][j] = (float)(rand()) / (float)RAND_MAX;
				}
			}
		}
	}

	// biasData
	float *biasData = (float *)malloc((size_t)numFilters * sizeof(float));
	for (int f = 0; f < numFilters; f++){
		biasData[f] = (float)(rand()) / (float)RAND_MAX;
	}

	// image (allocate as float *** to match convolution signature)
	int image_size = 5*scale;
	float ***image = (float ***)malloc((size_t)numChannels * sizeof(float**));
	for (int c = 0; c < numChannels; c++){
		image[c] = (float **)malloc((size_t)image_size * sizeof(float*));
		for (int i = 0; i < image_size; i++){
			image[c][i] = (float *)malloc((size_t)image_size * sizeof(float));
			for (int j = 0; j < image_size; j++){
				image[c][i][j] = (float)(rand()) / (float)RAND_MAX;
			}
		}
	}

	// seed rand
	srand((unsigned int)time(NULL));

	// call convolution
	float ***output = convolution(image, numChannels, kernel, biasData, numFilters, image_size, kernel_size);

	// // print one output value to verify
	int outSize = image_size - kernel_size + 1;
	// if (outSize > 0) {
	// 	printf("output[0][0][0] = %f\n", output[0][0][0]);
	// }

	// free output (the conv implementation allocated it); free what conv returned
	for (int f = 0; f < numFilters; f++){
		for (int i = 0; i < outSize; i++){
			free(output[f][i]);
		}
		free(output[f]);
	}
	free(output);

	// free image
	for (int c = 0; c < numChannels; c++){
		for (int i = 0; i < image_size; i++){
			free(image[c][i]);
		}
		free(image[c]);
	}
	free(image);

	// free kernel
	for (int f = 0; f < numFilters; f++){
		for (int c = 0; c < numChannels; c++){
			for (int i = 0; i < kernel_size; i++){
				free(kernel[f][c][i]);
			}
			free(kernel[f][c]);
		}
		free(kernel[f]);
	}
	free(kernel);

	free(biasData);

	return 0;
}
