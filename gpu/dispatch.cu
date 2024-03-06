#include <iostream>
#include <cuda_runtime.h>

#define N 3  // Size of the square matrices

// CUDA kernel to multiply two matrices
__global__ void matrixMul(float *a, float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < n && col < n) {
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int size = N * N * sizeof(float);

    // Allocate memory for host matrices
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);

    // Initialize host matrices
    for (int i = 0; i < N * N; ++i) {
        a[i] = i + 1;
        b[i] = i + 1;
    }

    // Allocate memory for device matrices
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy host matrices to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimGrid(1, 1);
    dim3 dimBlock(N, N);

    // Launch the kernel
    matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print result matrix
    for (int i = 0; i < N * N; ++i) {
        std::cout << c[i] << " ";
        if ((i + 1) % N == 0) {
            std::cout << std::endl;
        }
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}

