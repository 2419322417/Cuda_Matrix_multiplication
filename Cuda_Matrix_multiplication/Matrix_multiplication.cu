#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <math.h>

//Kernel函数，执行矩阵乘法
__global__ void Matrix_MulKernel(int m, int n, int k, float* A, float* B, float* C) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < m && Col < k) {
        float Cvalue = 0.0f;
        for (int i = 0; i < n; i++) {
            Cvalue += A[Row * n + i] * B[i * k + Col];
        }
        C[Row * k + Col] = Cvalue;
    }
}

//CPU版本矩阵乘法，用于验证结果
void Matrix_MulCPU(int m, int n, int k, float* A, float* B, float* C) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            float value = 0.0f;
            for (int i = 0; i < n; i++) {
                value += A[row * n + i] * B[i * k + col];
            }
            C[row * k + col] = value;
        }
    }
}

int Compare_Results(int m, int k, float* C_cpu, float* C_gpu) {
    float eps = 1e-5f;
    for (int i = 0; i < m * k; i++) {
        if (fabsf(C_cpu[i] - C_gpu[i]) > eps) {
            return 0; 
        }
    }
    return 1; 
}

int main() {
    int m = 4; 
    int n = 3; 
    int k = 5; 

	//分配主机内存
    size_t size_A = m * n * sizeof(float);
    size_t size_B = n * k * sizeof(float);
    size_t size_C = m * k * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    float* h_C_cpu = (float*)malloc(size_C);

	//初始化矩阵A和B,同时验证结果每个元素都为2.0f*n=6.0f
    for (int i = 0; i < m * n; i++) h_A[i] = 1.0f; 
    for (int i = 0; i < n * k; i++) h_B[i] = 2.0f;

	//分配设备内存
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((k + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    Matrix_MulKernel <<< numBlocks, threadsPerBlock >>> (m, n, k, d_A, d_B, d_C);
	//同步等待GPU完成
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    printf(" C (m=%d, k=%d):\n", m, k);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            printf("%5.1f ", h_C[i * k + j]);
        }
        printf("\n");
    }

    //CPU计算
    Matrix_MulCPU(m, n, k, h_A, h_B, h_C_cpu);

    //验证结果
    if (Compare_Results(m, k, h_C_cpu, h_C)) {
        printf(" \nCPU and GPU results match.\n");
    }
    else {
        printf(" \nCPU and GPU results differ.\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);

    return 0;
}
