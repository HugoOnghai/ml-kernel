#include "attention.h"
#include "functional.h"
#include <stdio.h>
#include <math.h>

// Scaled dot-product attention
float **scaled_dot_product_attention(float **Q, float **K, float **V, int seqLength, int depth)
{
    // TODO: Implement the attention algorithm
    /*
    Notes on SDPA from Attention Is All You Need
    
    Input:
    - queries packed into a matrix Q and keys packed into matrix K and values packed into matrix V of dimension dk and values of dimension dv
    - Q, K, and V for us are all of dimensions seqLength x depth.
    Steps:
    - compute the dot products of the query with all keys
    - divide each by sqrt(d_k) 
    - apply softmax function to obtain the weights on the values

    Attention(Q,K,V) = softmax( Q * K^T / sqrt(d_k) )*V
    */

    // ### multiply Q and K transpose (i.e. dot products of the query with all keys)
    // allocate memory for K_T
    float **K_T = (float**)malloc(depth*sizeof(float*));
    for (int i = 0; i < depth; i++)
    {
        K_T[i] = (float*)malloc(seqLength*sizeof(float));
    }

    for (int i = 0; i < depth; i++)
    {
        for (int j = 0; j < seqLength; j++)
        {
            K_T[i][j] = K[j][i];
        }
    }

    // apply dot products
    float** Q_KT = matmul(Q, K_T, seqLength, depth, depth, seqLength);

    // divide each by sqrt(d_k)
    float sqrtdepth = sqrtf(depth);
    for (int i = 0; i < seqLength; i++)
    {
        for (int j = 0; j < seqLength; j++)
        {
            Q_KT[i][j] = Q_KT[i][j] / sqrtdepth;
        }
    }

    // apply softmax function
    // my softmax function is for float* arrays, so operate row-wise (inputsize=seqLength)
    for (int i = 0; i < seqLength; i++)
    {
        Q_KT[i] = softmax(Q_KT[i], seqLength);
        for (int j = 0; j < seqLength; j++) // since my softmax function gives log-softmax, exponentiate each element
        {
            Q_KT[i][j] = expf(Q_KT[i][j]);
        }
    }

    // multiply softmax(QdotK^T) with V
    float** softmaxQKT_V = matmul(Q_KT, V, seqLength, seqLength, seqLength, depth);

    return softmaxQKT_V;
}