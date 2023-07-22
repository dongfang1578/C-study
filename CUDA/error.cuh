#ifndef ERROR_CUH_
#define ERROR_CUH_
#include <stdio.h>

#define CHECK(call)                                                     \
do                                                                      \
{                                                                       \
    const cudaError_t error_code = call;                                \
    if (error_code != cudaSuccess)                                      \
    {                                                                   \
        printf("CUDA Error:\n");                                        \
        printf("    File:   %s\n", __FILE__);                           \
        printf("    Line:   %s\n", __LINE__);                           \
        printf("    Error code: %d\n", error_code);                     \
        printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
        exit(1);                                                        \
    }                                                                   \
}while(0) 
#endif                                                              