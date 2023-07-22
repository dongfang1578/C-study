#include <stdio.h>

// 核函数
__global__ void helloFromGPU(void)
{
    printf("Hello World from GPU!\n");
}

__global__ void helloFromGPU1(void)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("Hello World from block %d and thread %d!\n", bid, tid);
}

__global__ void helloFromGPU2(void)
{
    const int b = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("Hello World from block-%d and thread-(%d, %d)!\n", b, tx, ty);
}

int main(void)
{
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();          // <<<grid, block>>>
    cudaDeviceReset();                  // 同步主机与设备
    return 0;
}