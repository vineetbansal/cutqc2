extern "C" __global__ void matrixAdd(const float* A, const float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = row * cols + col;
    if (row < rows && col < cols) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" __global__ void matrixSubtract(const float* A, const float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = row * cols + col;
    if (row < rows && col < cols) {
        C[idx] = A[idx] - B[idx];
    }
}