#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "../kernel/functional.h"

// test relu()

void main() {

    // profile relu operating on an array of values
    int num_elems = 1e6;
    float *array = (float *)malloc((size_t)num_elems * sizeof(float));
    for (int i = 0; i < num_elems; i++) {
        array[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f; // random values between -10 and 10
    }

    // printf("before: %f %f %f %f %f\n", array[0], array[1], array[2], array[3], array[4]);

    for (int i = 0; i < num_elems; i++) {
        array[i] = relu(array[i]);
    }

    // printf("after: %f %f %f %f %f\n", array[0], array[1], array[2], array[3], array[4]);

    free(array);
}