#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 16 
#define STREAMS 4       

//Kernel函数，执行矩阵乘法
__global__ void Matrix_MulKernel(int m, int n, int k, float* A, float* B, float* C) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < m && Col < k) {
        float cvalue = 0.0f;
        for (int i = 0; i < n; i++) {
            cvalue += A[Row * n + i] * B[i * k + Col];
        }
        C[Row * k + Col] = cvalue;
    }
}

//第一种优化，使用共享内存和平铺算法优化的矩阵乘法Kernel
__global__ void Matrix_MulKernel_Tiled(int m, int n, int k, float* A, float* B, float* C) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    float Cvalue = 0.0f;

    for (int t = 0; t < (n - 1) / TILE_WIDTH + 1; ++t) {
        // 加载A的tile
        if (Row < m && t * TILE_WIDTH + threadIdx.x < n) {
            ds_A[threadIdx.y][threadIdx.x] = A[Row * n + t * TILE_WIDTH + threadIdx.x];
        }
        else {
            ds_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 加载B的tile
        if (t * TILE_WIDTH + threadIdx.y < n && Col < k) {
            ds_B[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * k + Col];
        }
        else {
            ds_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 累积内积
        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (Row < m && Col < k) {
        C[Row * k + Col] = Cvalue;
    }
}

//第二种优化，在共享内存中添加padding，避免Bank Conflict的Kernel
__global__ void Matrix_MulKernel_Tiled_Padding(int m, int n, int k, float* A, float* B, float* C) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH + 1];  // +1避免Bank Conflict
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH + 1];

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    float Cvalue = 0.0f;

    for (int t = 0; t < (n - 1) / TILE_WIDTH + 1; ++t) {
        if (Row < m && t * TILE_WIDTH + threadIdx.x < n) {
            ds_A[threadIdx.y][threadIdx.x] = A[Row * n + t * TILE_WIDTH + threadIdx.x];
        }
        else {
            ds_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (t * TILE_WIDTH + threadIdx.y < n && Col < k) {
            ds_B[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * k + Col];
        }
        else {
            ds_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (Row < m && Col < k) {
        C[Row * k + Col] = Cvalue;
    }
}

//第三种优化，直接调整block大小。

//第四种优化，寄存器优化
__global__ void Matrix_MulKernel_RegTiling(int m, int n, int k, float* A, float* B, float* C) {
    __shared__ float d_B[TILE_WIDTH][TILE_WIDTH];

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    float Cvalue = 0.0f;

    for (int t = 0; t < (n - 1) / TILE_WIDTH + 1; ++t) {
        // 加载 B 的 tile 到共享内存
        if (t * TILE_WIDTH + threadIdx.y < n && Col < k) {
            d_B[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * k + Col];
        }
        else {
            d_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        #pragma unroll//展开循环以减少控制开销
        for (int i = 0; i < TILE_WIDTH; ++i) {
            int a_col = t * TILE_WIDTH + i;
            float a = 0.0f;
            if (Row < m && a_col < n) {
                a = A[Row * n + a_col];
            }
            Cvalue += a * d_B[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (Row < m && Col < k) {
        C[Row * k + Col] = Cvalue;
    }
}

//第五种优化，使用流（Streams）实现重叠计算和数据传输。
void Matrix_Mul_Overlapping(int m, int n, int k, float* h_A, float* h_B, float* h_C) {
    int rowsPerStream = m / STREAMS;
    int remainder = m % STREAMS;

    float* d_A[STREAMS];
    float* d_C[STREAMS];
    cudaStream_t streams[STREAMS];

    // 只分配一个 d_B
    float* d_B;
    size_t size_B = n * k * sizeof(float);
    cudaMalloc(&d_B, size_B);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    for (int i = 0; i < STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
        int thisRows = rowsPerStream + (i < remainder ? 1 : 0);
        if (thisRows == 0) continue;

        size_t size_A = (size_t)thisRows * (size_t)n * sizeof(float);
        size_t size_C = (size_t)thisRows * (size_t)k * sizeof(float);
        cudaMalloc(&d_A[i], size_A);
        cudaMalloc(&d_C[i], size_C);
    }

    dim3 threads(TILE_WIDTH, TILE_WIDTH);

    int rowOffset = 0;
    for (int i = 0; i < STREAMS; ++i) {
        int thisRows = rowsPerStream + (i < remainder ? 1 : 0);
        if (thisRows == 0) continue;

        size_t size_A = thisRows * n * sizeof(float);
        size_t size_C = thisRows * k * sizeof(float);

        // 分配 grid：注意 block 数基于 thisRows
        dim3 blocks((k + TILE_WIDTH - 1) / TILE_WIDTH, (thisRows + TILE_WIDTH - 1) / TILE_WIDTH);

        // 异步拷贝 A 片段到 device
        cudaMemcpyAsync(d_A[i], h_A + (size_t)rowOffset * (size_t)n, size_A, cudaMemcpyHostToDevice, streams[i]);

        // kernel：传入 thisRows 作为 m 参数（kernel 内会以 Row < m 做边界检查）
        Matrix_MulKernel_Tiled <<< blocks, threads, 0, streams[i] >>> (thisRows, n, k, d_A[i], d_B, d_C[i]);

        // 异步拷回 C 片段
        cudaMemcpyAsync(h_C + (size_t)rowOffset * (size_t)k, d_C[i], size_C, cudaMemcpyDeviceToHost, streams[i]);

        rowOffset += thisRows;
    }

    // 等待所有 stream 完成
    cudaDeviceSynchronize();

    // 清理
    for (int i = 0; i < STREAMS; ++i) {
        cudaFree(d_A[i]);
        cudaFree(d_C[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_B);
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
    int m = 1024; 
    int n = 1024; 
    int k = 1024; 

	//分配主机内存
    size_t size_A = m * n * sizeof(float);
    size_t size_B = n * k * sizeof(float);
    size_t size_C = m * k * sizeof(float);

    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    float* h_C_cpu = (float*)malloc(size_C);
    
     // 普通 malloc（用于前四种优化方法）
     h_A = (float*)malloc(size_A);
     h_B = (float*)malloc(size_B);
     h_C = (float*)malloc(size_C);

     // pinned memory
     /*cudaHostAlloc((void**)&h_A, size_A, cudaHostAllocDefault);
     cudaHostAlloc((void**)&h_B, size_B, cudaHostAllocDefault);
     cudaHostAlloc((void**)&h_C, size_C, cudaHostAllocDefault);*/

	//初始化矩阵A和B,同时验证结果每个元素都为2.0f*n
    for (int i = 0; i < m * n; i++) h_A[i] = 1.0f; 
    for (int i = 0; i < n * k; i++) h_B[i] = 2.0f;
    for (int i = 0; i < m * k; i++) h_C[i] = 0.0f;

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

   /* dim3 threadsPerBlock_tiled(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks_tiled((k + TILE_WIDTH - 1) / TILE_WIDTH,
        (m + TILE_WIDTH - 1) / TILE_WIDTH);
    Matrix_MulKernel_Tiled <<< numBlocks_tiled, threadsPerBlock_tiled >>> (m, n, k, d_A, d_B, d_C);
    Matrix_MulKernel_Tiled_Padding<<<numBlocks_tiled, threadsPerBlock_tiled >>>(m, n, k, d_A, d_B, d_C);
    Matrix_MulKernel_RegTiling<<<numBlocks_tiled, threadsPerBlock_tiled>>>(m, n, k, d_A, d_B, d_C);
    Matrix_Mul_Overlapping(m, n, k, h_A, h_B, h_C);*/

	//同步等待GPU完成
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    //printf(" C (m=%d, k=%d):\n", m, k);
    //for (int i = 0; i < m; i++) {
    //    for (int j = 0; j < k; j++) {
    //        printf("%5.1f ", h_C[i * k + j]);
    //    }
    //    printf("\n");
    //}

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

    free(h_C_cpu);
    //前四种优化方法
    free(h_A);
    free(h_B);
    free(h_C);

    //对应pinned释放

    //cudaFreeHost(h_A);
    //cudaFreeHost(h_B);
    //cudaFreeHost(h_C);

    return 0;
}
