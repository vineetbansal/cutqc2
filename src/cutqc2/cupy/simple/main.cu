#include "simple.cu"
#include <iostream>
#include <cuda_runtime.h>


int main() {
    const int rows = 4;
    const int cols = 4;
    const int size = rows * cols * sizeof(float);

    float h_A[rows * cols], h_B[rows * cols], h_C[rows * cols];
    for (int i = 0; i < rows * cols; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result matrix C:\n";
    for (int i = 0; i < rows * cols; ++i) {
        std::cout << h_C[i] << " ";
        if ((i + 1) % cols == 0) std::cout << "\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
