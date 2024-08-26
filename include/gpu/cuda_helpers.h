#ifndef MPS_PROJECT_CUDA_HELPERS_H_
#define MPS_PROJECT_CUDA_HELPERS_H_

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cstdio>

/**
 * @brief Macro to check for CUDA errors.
 *
 * This macro wraps CUDA runtime API calls and checks if they return an error.
 * If an error occurs, it prints an error message with the file name and line number,
 * and returns `-1` to indicate failure.
 *
 * @param err The CUDA runtime API call.
 */
#define CUDA_CHECK(err) {                                 \
    cudaError_t errCode = err;                            \
    if (errCode != cudaSuccess) {                         \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",      \
                cudaGetErrorString(errCode),              \
                __FILE__, __LINE__);                      \
        return -1;                                        \
    }                                                     \
}

/**
 * @brief Macro to check for CUDA kernel launch errors.
 *
 * This macro wraps CUDA kernel calls and checks for any errors during launch.
 * It uses `cudaGetLastError` to get the last error and reports it.
 *
 * @param func The CUDA kernel call.
 */
#define CUDA_KERNEL_CHECK(func) {                         \
    func;                                                 \
    cudaError_t errCode = cudaGetLastError();             \
    if (errCode != cudaSuccess) {                         \
        fprintf(stderr, "CUDA Kernel Error: %s at %s:%d\n", \
                cudaGetErrorString(errCode),              \
                __FILE__, __LINE__);                      \
        return -1;                                        \
    }                                                     \
}

#endif  // MPS_PROJECT_CUDA_HELPERS_H_
