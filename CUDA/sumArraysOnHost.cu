#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

__global__ void sumArrayOnGPU(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned int)time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand()&0xFF) / 10.0f;
    }
}

void showData(float *ip, int size)
{
    for (int i = 0; i < size / 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            printf("%10.3f ", ip[i*8+j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    int nElem = 24;
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);
    printf("h_A:\n");
    showData(h_A, nElem);
    printf("h_B:\n");
    showData(h_B, nElem);

    // 在GPU上申请内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // 把数据从主机内存拷贝到GPU的全局内存
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    sumArraysOnHost(h_A, h_B, h_C, nElem);

    // 把结果从GPU复制回到主机的数组h_C中
    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);
    printf("h_C:\n");
    showData(h_C, nElem);

    // 释放GPU内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}