#include <cnn_module.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <iostream>

// 错误检查宏
#define CUBLAS_CHECK(func) do { \
    cublasStatus_t status = func; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error: %d (func: %s, line: %d)\n", status, __func__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUDA_CHECK(func) do { \
    cudaError_t status = func; \
    if (status != cudaSuccess) { \
        printf("CUDA Error: %s (func: %s, line: %d)\n", cudaGetErrorString(status), __func__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CURAND_CHECK(func) do { \
    curandStatus_t status = func; \
    if (status != CURAND_STATUS_SUCCESS) { \
        printf("cuRAND Error: %d (func: %s, line: %d)\n", status, __func__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static cublasHandle_t cublas_handle;
curandGenerator_t curand_generator;

// 1. CUDA 核函数（GPU 版 ReLu 正向）
__global__ void relu_kernel(const float* d_input, float* d_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = (d_input[idx] > 0.0f) ? d_input[idx] : 0.0f;
    }
}

// 2. GPU 版 ReLu 正向函数（对外接口和 CPU 版一致）
void forward_relu_gpu(const float* d_input, float* d_output, int size) {
    // 校验 GPU 指针（简单判断，实际可加 CUDA 错误检查）
    if (!d_input || !d_output || size <= 0) {
        throw std::invalid_argument("GPU ReLU 输入无效");
    }

    // 配置核函数网格/块大小
    dim3 block(256);  // 线程块大小（常用 256/512）
    dim3 grid((size + block.x - 1) / block.x);  // 网格大小（覆盖所有元素）

    // 启动核函数
    relu_kernel<<<grid, block>>>(d_input, d_output, size);

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("GPU ReLU 正向错误：" + std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();  // 等待核函数执行完成
}

// 3. GPU 版 ReLu 反向核函数
__global__ void relu_backward_kernel(const float* d_input, const float* d_grad_output, float* d_grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_grad_input[idx] = (d_input[idx] > 0.0f) ? d_grad_output[idx] : 0.0f;
    }
}

// 4. GPU 版 ReLu 反向函数
void backward_relu_gpu(const float* d_input, const float* d_grad_output, float* d_grad_input, int size) {
    if (!d_input || !d_grad_output || !d_grad_input || size <= 0) {
        throw std::invalid_argument("GPU ReLU 反向输入无效");
    }

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    relu_backward_kernel<<<grid, block>>>(d_input, d_grad_output, d_grad_input, size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("GPU ReLU 反向错误：" + std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();
}
// --------------------------
// 1. CUDA 核函数：Sigmoid 正向（GPU 设备端）
// --------------------------
__global__ void sigmoid_forward_kernel(const float* d_input, float* d_output, int size) {
    // 计算当前线程索引（1D 线程布局，适配任意尺寸）
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;  // 避免越界（线程数可能大于元素数）

    // 稳定版 Sigmoid 计算（用 tanh 避免 exp 溢出）
    float x = d_input[idx];
    d_output[idx] = 0.5f * (1.0f + tanhf(0.5f * x));
}

// --------------------------
// 2. 主机端正向函数（对外接口）
// --------------------------
void forward_sigmoid_gpu(const float* d_input, float* d_output, int size) {
    // 入参校验（GPU 指针非空 + 尺寸合法）
    if (!d_input || !d_output || size <= 0) {
        throw std::invalid_argument("Sigmoid 正向：输入/输出指针为空或尺寸无效");
    }

    // 配置 CUDA 线程布局（常用 256 线程/块，兼顾效率和兼容性）
    dim3 block_size(256);
    // 网格大小 = 向上取整（总元素数 / 线程块大小），确保覆盖所有元素
    dim3 grid_size((size + block_size.x - 1) / block_size.x);

    // 启动 CUDA 核函数（<<<网格大小, 块大小>>>）
    sigmoid_forward_kernel<<<grid_size, block_size>>>(d_input, d_output, size);

    // 检查核函数启动错误（关键：避免静默失败）
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Sigmoid 正向核函数启动失败：" + std::string(cudaGetErrorString(err)));
    }

    // 等待核函数执行完成（确保后续操作前数据已计算完毕）
    cudaDeviceSynchronize();
}

// --------------------------
// 3. CUDA 核函数：Sigmoid 反向（GPU 设备端）
// --------------------------
__global__ void sigmoid_backward_kernel(const float* d_output, const float* d_grad_output, float* d_grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // 梯度计算：grad_input = grad_output * output * (1 - output)
    float out = d_output[idx];
    d_grad_input[idx] = d_grad_output[idx] * out * (1.0f - out);
}

// --------------------------
// 4. 主机端反向函数（对外接口）
// --------------------------
void backward_sigmoid_gpu(const float* d_output, const float* d_grad_output, float* d_grad_input, int size) {
    // 入参校验
    if (!d_output || !d_grad_output || !d_grad_input || size <= 0) {
        throw std::invalid_argument("Sigmoid 反向：输入/输出指针为空或尺寸无效");
    }

    // 线程布局（和正向一致，保持统一）
    dim3 block_size(256);
    dim3 grid_size((size + block_size.x - 1) / block_size.x);

    // 启动反向核函数
    sigmoid_backward_kernel<<<grid_size, block_size>>>(d_output, d_grad_output, d_grad_input, size);

    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Sigmoid 反向核函数启动失败：" + std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();
}
__global__ void flatten_kernel(
    const float* weights, 
    float* weights_flat, 
    int C_out, 
    int C_in, 
    int kernel_size, 
    int col_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C_out * col_height) return;
    int c_out = idx / col_height;
    int c_in_k = idx % col_height;
    int c_in = c_in_k % C_in;
    int k_pos = c_in_k / C_in;
    int kh = k_pos / kernel_size;
    int kw = k_pos % kernel_size;
    int weight_idx = c_out * C_in * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw;
    weights_flat[c_out + c_in_k * C_out] = weights[weight_idx];
}

// 2. 卷积层：特征图重塑（[C_out, N×H×W] → [N, C_out, H, W]）
__global__ void reshape_kernel(
    const float* conv_out, 
    float* output, 
    int C_out, 
    int N, 
    int H, 
    int W, 
    int col_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C_out * col_width) return;
    int c_out = idx / col_width;
    int col_col = idx % col_width;
    int n = col_col / (H * W);
    int spatial_idx = col_col % (H * W);
    int h = spatial_idx / W;
    int w = spatial_idx % W;
    int out_idx = n * C_out * H * W + c_out * H * W + h * W + w;
    output[out_idx] = conv_out[c_out + col_col * C_out];
}

// 3. 卷积层反向：梯度扁平（[N, C_out, H, W] → [C_out, N×H×W]）
__global__ void flat_grad_kernel(
    const float* grad_output, 
    float* grad_output_flat, 
    int C_out, 
    int N, 
    int H, 
    int W, 
    int col_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C_out * col_width) return;
    int c_out = idx / col_width;
    int col_col = idx % col_width;
    int n = col_col / (H * W);
    int spatial_idx = col_col % (H * W);
    int h = spatial_idx / W;
    int w = spatial_idx % W;
    int grad_idx = n * C_out * H * W + c_out * H * W + h * W + w;
    grad_output_flat[c_out + col_col * C_out] = grad_output[grad_idx];
}

// 4. 卷积层反向：权重梯度重塑（[C_out, C_in×K×K] → [C_out, C_in, K, K]）
__global__ void reshape_weight_kernel(
    const float* grad_weights_flat, 
    float* grad_weights, 
    int C_out, 
    int C_in, 
    int kernel_size, 
    int col_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C_out * col_height) return;
    int c_out = idx / col_height;
    int c_in_k = idx % col_height;
    int c_in = c_in_k % C_in;
    int k_pos = c_in_k / C_in;
    int kh = k_pos / kernel_size;
    int kw = k_pos % kernel_size;
    int grad_idx = c_out * C_in * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw;
    // grad_weights_flat is column-major (rows=C_out, cols=col_height)
    grad_weights[grad_idx] = grad_weights_flat[c_out + c_in_k * C_out];
}

// 5. 池化层正向：2×2 Max Pooling
__global__ void pool_kernel(
    const float* input, 
    float* output, 
    float* mask, 
    int N, 
    int C, 
    int H, 
    int W, 
    int out_H, 
    int out_W
) {
    int n = blockIdx.z;
    int c = blockIdx.y;
    int oh = blockIdx.x * blockDim.y + threadIdx.y;
    int ow = threadIdx.x;
    if (oh >= out_H || ow >= out_W) return;

    int ih_start = oh * 2;
    int iw_start = ow * 2;
    float max_val = -1e9f;
    int max_ih = 0, max_iw = 0;
    for (int kh = 0; kh < 2; kh++) {
        for (int kw = 0; kw < 2; kw++) {
            int ih = ih_start + kh;
            int iw = iw_start + kw;
            int in_idx = n * C * H * W + c * H * W + ih * W + iw;
            float val = input[in_idx];
            if (val > max_val) {
                max_val = val;
                max_ih = ih;
                max_iw = iw;
            }
        }
    }

    int out_idx = n * C * out_H * out_W + c * out_H * out_W + oh * out_W + ow;
    output[out_idx] = max_val;

    int mask_base = n * C * H * W + c * H * W;
    for (int kh = 0; kh < 2; kh++) {
        for (int kw = 0; kw < 2; kw++) {
            int ih = ih_start + kh;
            int iw = iw_start + kw;
            mask[mask_base + ih * W + iw] = 0.0f;
        }
    }
    mask[mask_base + max_ih * W + max_iw] = 1.0f;
}

// 6. 池化层反向：梯度回传（仅最大值位置）
__global__ void pool_grad_kernel(
    const float* grad_output, 
    const float* mask, 
    float* grad_input, 
    int N, 
    int C, 
    int H, 
    int W, 
    int out_H, 
    int out_W
) {
    int n = blockIdx.z;
    int c = blockIdx.y;
    int ih = blockIdx.x * blockDim.y + threadIdx.y;
    int iw = threadIdx.x;
    if (ih >= H || iw >= W) return;

    int oh = ih / 2;
    int ow = iw / 2;
    int mask_idx = n * C * H * W + c * H * W + ih * W + iw;
    int out_idx = n * C * out_H * out_W + c * out_H * out_W + oh * out_W + ow;
    grad_input[mask_idx] = grad_output[out_idx] * mask[mask_idx];
}

// 7. SoftMax层：计算每行最大值
__global__ void softmax_max_kernel(
    const float* input, 
    float* row_max, 
    int batch_size, 
    int features
) {
    int n = blockIdx.x;
    if (n >= batch_size) return;
    float max_val = -1e9f;
    for (int c = 0; c < features; c++) {
        int idx = n * features + c;
        max_val = fmaxf(max_val, input[idx]);
    }
    row_max[n] = max_val;
}

// 8. SoftMax层：输入减最大值+指数运算
__global__ void softmax_exp_kernel(
    const float* input, 
    const float* row_max, 
    float* output, 
    int batch_size, 
    int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;
    int n = idx / features;
    output[idx] = expf(input[idx] - row_max[n]);
}

// 9. SoftMax层：计算每行和
__global__ void softmax_sum_kernel(
    const float* output, 
    float* row_sum, 
    int batch_size, 
    int features
) {
    int n = blockIdx.x;
    if (n >= batch_size) return;
    float sum_val = 0.0f;
    for (int c = 0; c < features; c++) {
        sum_val += output[n * features + c];
    }
    row_sum[n] = sum_val;
}

// 10. SoftMax层：归一化（概率）
__global__ void softmax_norm_kernel(
    float* output, 
    const float* row_sum, 
    int batch_size, 
    int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;
    int n = idx / features;
    output[idx] /= (row_sum[n] + 1e-8f); // 避免除零
}

// 11. Cross Entropy：计算每个样本损失
__global__ void ce_loss_kernel(
    const float* softmax_out, 
    const int* labels, 
    float* per_sample_loss, 
    int batch_size, 
    int features
) {
    int n = blockIdx.x;
    if (n >= batch_size) return;
    int c = labels[n];
    int idx = n * features + c;
    per_sample_loss[n] = -logf(softmax_out[idx] + 1e-8f); // 避免log(0)
}

// 12. Cross Entropy：计算logits梯度
__global__ void ce_grad_kernel(
    const float* softmax_out, 
    const int* labels, 
    float* grad_logits, 
    float scale, 
    int batch_size, 
    int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;
    int n = idx / features;
    int c = idx % features;
    int label = labels[n];
    float one_hot = (c == label) ? 1.0f : 0.0f;
    grad_logits[idx] = (softmax_out[idx] - one_hot) * scale;
}

// ========================== 工具函数 ==========================
// void init_cublas() {
//     CUBLAS_CHECK(cublasCreate(&cublas_handle));
// }
// void destroy_cublas() {
//     CUBLAS_CHECK(cublasDestroy(cublas_handle));
// }
// 在 cnn_module.cu 中找到这两个函数并替换

void init_cublas() {
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CURAND_CHECK(curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator, 1234ULL));
}

void destroy_cublas() {
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CURAND_CHECK(curandDestroyGenerator(curand_generator));
}
void create_ones_gpu(float** ones, int rows, int cols) {
    size_t size = rows * cols * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)ones, size));
    float* ones_cpu = (float*)malloc(size);
    for (int i = 0; i < rows * cols; i++) ones_cpu[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(*ones, ones_cpu, size, cudaMemcpyHostToDevice));
    free(ones_cpu);
}

// 初始化GPU权重（正态分布）
void init_weights_gpu(float** weights, int rows, int cols) {
    size_t size = rows * cols * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)weights, size));  
    CURAND_CHECK(curandGenerateNormal(curand_generator, *weights, rows * cols, 0.0f, 0.01f));
}

// 初始化GPU偏置（0值）
void init_bias_gpu(float** bias, int size) {
    CUDA_CHECK(cudaMalloc((void**)bias, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(*bias, 0, size * sizeof(float)));
}

// ========================== 全连接层 ==========================
// 前向声明 FC 内核（确保在使用处已声明）
__global__ void fc_forward_kernel(const float* input, const float* weights, const float* bias, float* output,
                                  int batch_size, int in_features, int out_features);
__global__ void fc_grad_input_kernel(const float* grad_output, const float* weights, float* grad_input,
                                     int batch_size, int in_features, int out_features);
__global__ void fc_grad_weights_kernel(const float* grad_output, const float* input, float* grad_weights,
                                      int batch_size, int in_features, int out_features);
__global__ void fc_grad_bias_kernel(const float* grad_output, float* grad_bias, int batch_size, int out_features);

void forward_fc(float* input, float* output, float* weights, float* bias,
                int batch_size, int in_features, int out_features) {
    // 使用直接的 CUDA 核函数按行主 (row-major) 存储的数组计算
    // input: [batch_size, in_features]  row-major
    // weights: [out_features, in_features] row-major (每行是一个输出通道的权重)
    // output: [batch_size, out_features] row-major

    // 每个线程计算一个 output 元素 (n, m)
    dim3 block(16, 16);
    dim3 grid((batch_size + block.x - 1) / block.x, (out_features + block.y - 1) / block.y);
    fc_forward_kernel<<<grid, block>>>(input, weights, bias, output, batch_size, in_features, out_features);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void backward_fc(float* input, float* output, float* weights, float* bias,
                 int batch_size, int in_features, int out_features,
                 float* grad_output, float* grad_input, float* grad_weights,
                 float* grad_bias) {
    // 使用直接 CUDA 核函数逐元素计算梯度，避免 cuBLAS 行/列主布局歧义
    dim3 block_fw(16, 16);
    // grad_input kernel: grid over (batch_size, in_features)
    dim3 grid_gin((batch_size + block_fw.x - 1) / block_fw.x, (in_features + block_fw.y - 1) / block_fw.y);
    fc_grad_input_kernel<<<grid_gin, block_fw>>>(grad_output, weights, grad_input, batch_size, in_features, out_features);

    // grad_weights kernel: grid over (out_features, in_features)
    dim3 grid_gw((out_features + block_fw.x - 1) / block_fw.x, (in_features + block_fw.y - 1) / block_fw.y);
    fc_grad_weights_kernel<<<grid_gw, block_fw>>>(grad_output, input, grad_weights, batch_size, in_features, out_features);

    // grad_bias kernel: one thread per out feature
    dim3 block_b(256);
    dim3 grid_b((out_features + block_b.x - 1) / block_b.x);
    fc_grad_bias_kernel<<<grid_b, block_b>>>(grad_output, grad_bias, batch_size, out_features);

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ========================== 卷积层 ==========================
// im2col：输入特征图→矩阵
__global__ void im2col_kernel(const float* input, float* col,
                              int N, int C_in, int H, int W, int kernel_size, int padding) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_height = C_in * kernel_size * kernel_size;
    int col_width = N * H * W;
    if (col_idx >= col_height * col_width) return;

    int c_col = col_idx / col_width;
    int col_col = col_idx % col_width;
    int n = col_col / (H * W);
    int spatial_idx = col_col % (H * W);
    int h = spatial_idx / W;
    int w = spatial_idx % W;

    int kernel_pos = c_col / C_in;
    int c = c_col % C_in;
    int kh = kernel_pos / kernel_size - padding;
    int kw = kernel_pos % kernel_size - padding;

    int h_in = h + kh;
    int w_in = w + kw;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        int input_idx = n * C_in * H * W + c * H * W + h_in * W + w_in;
        col[c_col + col_col * col_height] = input[input_idx];
    } else {
        col[c_col + col_col * col_height] = 0.0f;
    }
}

void im2col(float* input, float* col, int N, int C_in, int H, int W, int kernel_size, int padding) {
    int col_height = C_in * kernel_size * kernel_size;
    int col_width = N * H * W;
    int total_elements = col_height * col_width;

    dim3 block(1024);
    dim3 grid((total_elements + block.x - 1) / block.x);
    im2col_kernel<<<grid, block>>>(input, col, N, C_in, H, W, kernel_size, padding);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// col2im：矩阵→特征图
__global__ void col2im_kernel(const float* col, float* output,
                              int N, int C_in, int H, int W, int kernel_size, int padding) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_size = N * C_in * H * W;
    if (out_idx >= out_size) return;

    int n = out_idx / (C_in * H * W);
    int rem = out_idx % (C_in * H * W);
    int c = rem / (H * W);
    int h = (rem % (H * W)) / W;
    int w = (rem % (H * W)) % W;

    float sum = 0.0f;
    int col_height = C_in * kernel_size * kernel_size;
    int col_width = N * H * W;
    for (int kh = -padding; kh <= padding; kh++) {
        for (int kw = -padding; kw <= padding; kw++) {
            int h_col = h - kh;
            int w_col = w - kw;
            if (h_col < 0 || h_col >= H || w_col < 0 || w_col >= W) continue;

            int kernel_pos = (kh + padding) * kernel_size + (kw + padding);
            int c_col = kernel_pos * C_in + c;
            int col_col = n * H * W + h_col * W + w_col;
            int col_idx = c_col + col_col * col_height;
            sum += col[col_idx];
        }
    }
    output[out_idx] = sum;
}

void col2im(float* col, float* output, int N, int C_in, int H, int W, int kernel_size, int padding) {
    int out_size = N * C_in * H * W;
    dim3 block(1024);
    dim3 grid((out_size + block.x - 1) / block.x);
    col2im_kernel<<<grid, block>>>(col, output, N, C_in, H, W, kernel_size, padding);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 卷积层正向传播
void forward_conv(float* input, float* output, float* weights, float* bias,
                  int N, int C_in, int H, int W, int C_out) {
    const int kernel_size = 3;
    const int padding = 1;
    const int col_height = C_in * kernel_size * kernel_size;
    const int col_width = N * H * W;

    // 1. im2col转换
    float* col;
    CUDA_CHECK(cudaMalloc(&col, col_height * col_width * sizeof(float)));
    im2col(input, col, N, C_in, H, W, kernel_size, padding);

    // 2. 卷积核扁平
    float* weights_flat;
    CUDA_CHECK(cudaMalloc(&weights_flat, C_out * col_height * sizeof(float)));
    dim3 block_flat(1024);
    dim3 grid_flat((C_out * col_height + block_flat.x - 1) / block_flat.x);
    flatten_kernel<<<grid_flat, block_flat>>>(weights, weights_flat, C_out, C_in, kernel_size, col_height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. GEMM计算
    float* conv_out;
    CUDA_CHECK(cudaMalloc(&conv_out, C_out * col_width * sizeof(float)));
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        C_out, col_width, col_height,
        &alpha,
        weights_flat, C_out,
        col, col_height,
        &beta,
        conv_out, C_out
    ));

    // 4. 加偏置
    float* ones_gpu;
    create_ones_gpu(&ones_gpu, 1, col_width);
    const float alpha_bias = 1.0f, beta_bias = 1.0f;
    CUBLAS_CHECK(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        C_out, col_width, 1,
        &alpha_bias,
        bias, C_out,
        ones_gpu, 1,
        &beta_bias,
        conv_out, C_out
    ));

    // 5. 重塑为特征图
    dim3 block_reshape(1024);
    dim3 grid_reshape((C_out * col_width + block_reshape.x - 1) / block_reshape.x);
    reshape_kernel<<<grid_reshape, block_reshape>>>(conv_out, output, C_out, N, H, W, col_width);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(col));
    CUDA_CHECK(cudaFree(weights_flat));
    CUDA_CHECK(cudaFree(conv_out));
    CUDA_CHECK(cudaFree(ones_gpu));
}

// 卷积层反向传播
void backward_conv(float* input, float* weights, float* grad_output,
                   float* grad_input, float* grad_weights, float* grad_bias,
                   int N, int C_in, int H, int W, int C_out) {
    const int kernel_size = 3;
    const int padding = 1;
    const int col_height = C_in * kernel_size * kernel_size;
    const int col_width = N * H * W;
    const float scale = 1.0f;

    // 1. 偏置梯度
    float* ones_gpu;
    create_ones_gpu(&ones_gpu, N * H * W, 1);
    float* grad_output_flat;
    CUDA_CHECK(cudaMalloc(&grad_output_flat, C_out * col_width * sizeof(float)));   
    dim3 block_flat(1024);
    dim3 grid_flat((C_out * col_width + block_flat.x - 1) / block_flat.x);
    flat_grad_kernel<<<grid_flat, block_flat>>>(grad_output, grad_output_flat, C_out, N, H, W, col_width);
    CUDA_CHECK(cudaDeviceSynchronize());
    float beta = 0.0f;
    const float one_alpha = 1.0f;
    CUBLAS_CHECK(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        C_out, 1, col_width,
        &one_alpha,
        grad_output_flat, C_out,
        ones_gpu, col_width,
        &beta,
        grad_bias, C_out
    ));

    // 2. 权重梯度
    float* col;
    CUDA_CHECK(cudaMalloc(&col, col_height * col_width * sizeof(float)));
    im2col(input, col, N, C_in, H, W, kernel_size, padding);
    float* grad_weights_flat;
    CUDA_CHECK(cudaMalloc(&grad_weights_flat, C_out * col_height * sizeof(float)));
    CUBLAS_CHECK(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        C_out, col_height, col_width,
        &one_alpha,
        grad_output_flat, C_out,
        col, col_height,
        &beta,
        grad_weights_flat, C_out
    ));
    dim3 block_weight(1024);
    dim3 grid_weight((C_out * col_height + block_weight.x - 1) / block_weight.x);
    reshape_weight_kernel<<<grid_weight, block_weight>>>(grad_weights_flat, grad_weights, C_out, C_in, kernel_size, col_height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. 输入梯度
    float* weights_flat;
    CUDA_CHECK(cudaMalloc(&weights_flat, C_out * col_height * sizeof(float)));
    flatten_kernel<<<grid_weight, block_weight>>>(weights, weights_flat, C_out, C_in, kernel_size, col_height);
    CUDA_CHECK(cudaDeviceSynchronize());
    float* grad_col;
    CUDA_CHECK(cudaMalloc(&grad_col, col_height * col_width * sizeof(float)));
    float alpha = 1.0f;
    CUBLAS_CHECK(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        col_height, col_width, C_out,
        &alpha,
        weights_flat, C_out,
        grad_output_flat, C_out,
        &beta,
        grad_col, col_height
    ));
    col2im(grad_col, grad_input, N, C_in, H, W, kernel_size, padding);

    CUDA_CHECK(cudaFree(ones_gpu));
    CUDA_CHECK(cudaFree(grad_output_flat));
    CUDA_CHECK(cudaFree(col));
    CUDA_CHECK(cudaFree(grad_weights_flat));
    CUDA_CHECK(cudaFree(weights_flat));
    CUDA_CHECK(cudaFree(grad_col));
}

// ========================== 池化层 ==========================
void forward_max_pool(float* input, float* output, float* mask,
                      int N, int C, int H, int W) {
    assert(H % 2 == 0 && W % 2 == 0);
    int out_H = H / 2;
    int out_W = W / 2;

    dim3 block(32, 32);
    dim3 grid((out_H + block.y - 1) / block.y, C, N);
    pool_kernel<<<grid, block>>>(input, output, mask, N, C, H, W, out_H, out_W);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void backward_max_pool(float* grad_output, float* mask, float* grad_input,
                       int N, int C, int H, int W) {
    assert(H % 2 == 0 && W % 2 == 0);
    int out_H = H / 2;
    int out_W = W / 2;

    dim3 block(32, 32);
    dim3 grid((H + block.y - 1) / block.y, C, N);
    pool_grad_kernel<<<grid, block>>>(grad_output, mask, grad_input, N, C, H, W, out_H, out_W);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ========================== SoftMax层 ==========================
void stable_softmax(float* input, float* output, int batch_size, int features) {
    // 1. 计算每行最大值
    float* row_max;
    CUDA_CHECK(cudaMalloc(&row_max, batch_size * sizeof(float)));
    dim3 grid_max(batch_size);
    softmax_max_kernel<<<grid_max, 1>>>(input, row_max, batch_size, features);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. 输入减最大值+指数
    float* output_exp;
    CUDA_CHECK(cudaMalloc(&output_exp, batch_size * features * sizeof(float)));
    dim3 block_exp(1024);
    dim3 grid_exp((batch_size * features + block_exp.x - 1) / block_exp.x);
    softmax_exp_kernel<<<grid_exp, block_exp>>>(input, row_max, output_exp, batch_size, features);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. 计算每行和
    float* row_sum;
    CUDA_CHECK(cudaMalloc(&row_sum, batch_size * sizeof(float)));
    dim3 grid_sum(batch_size);
    softmax_sum_kernel<<<grid_sum, 1>>>(output_exp, row_sum, batch_size, features);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. 归一化
    softmax_norm_kernel<<<grid_exp, block_exp>>>(output_exp, row_sum, batch_size, features);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, output_exp, batch_size * features * sizeof(float), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(row_max));
    CUDA_CHECK(cudaFree(output_exp));
    CUDA_CHECK(cudaFree(row_sum));
}

// ========================== Cross Entropy Loss ==========================
void cross_entropy_softmax(float* logits, int* labels, float* loss, float* grad_logits,
                           int batch_size, int features) {
    // 1. 计算SoftMax概率
    float* softmax_out;
    CUDA_CHECK(cudaMalloc(&softmax_out, batch_size * features * sizeof(float)));
    stable_softmax(logits, softmax_out, batch_size, features);

    // 2. 计算每个样本损失
    // labels 可能是 host 内存（来自 Python 的 numpy 数组），不能直接在 device kernel 中访问
    // 所以先把 labels 拷贝到 device 上，然后在 kernel 中使用 device 指针
    int* d_labels = nullptr;
    // 检测传入的 labels 指针是否已经是 device 指针（例如用户可能传入 device memory）
    cudaPointerAttributes ptr_attr;
    cudaError_t attr_err = cudaPointerGetAttributes(&ptr_attr, (const void*)labels);
    // fprintf(stderr, "[DEBUG] cudaPointerGetAttributes returned %d\n", (int)attr_err);
    int labels_on_device = 0;
    if (attr_err == cudaSuccess) {
#if CUDART_VERSION >= 10000
        if (ptr_attr.type == cudaMemoryTypeDevice) labels_on_device = 1;
#else
        if (ptr_attr.memoryType == cudaMemoryTypeDevice) labels_on_device = 1;
#endif
    }

    if (labels_on_device) {
        // 直接使用设备指针（注意：这种用法要求调用方保证 labels 在 device 上）
        d_labels = labels;
    } else {
        // labels 在 host 上，需拷贝到 device
        CUDA_CHECK(cudaMalloc(&d_labels, batch_size * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_labels, labels, batch_size * sizeof(int), cudaMemcpyHostToDevice));
    }

    float* per_sample_loss;
    CUDA_CHECK(cudaMalloc(&per_sample_loss, batch_size * sizeof(float)));
    dim3 grid_loss(batch_size);
    ce_loss_kernel<<<grid_loss, 1>>>(softmax_out, d_labels, per_sample_loss, batch_size, features);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. 计算平均损失（CPU）
    float* per_sample_loss_cpu = (float*)malloc(batch_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(per_sample_loss_cpu, per_sample_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
    *loss = 0.0f;
    for (int n = 0; n < batch_size; n++) {
        *loss += per_sample_loss_cpu[n];
    }
    *loss /= batch_size;

    // 4. 计算logits梯度
    float scale = 1.0f / batch_size;
    dim3 block_grad(1024);
    dim3 grid_grad((batch_size * features + block_grad.x - 1) / block_grad.x);
    ce_grad_kernel<<<grid_grad, block_grad>>>(softmax_out, d_labels, grad_logits, scale, batch_size, features);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 释放临时内存
    CUDA_CHECK(cudaFree(softmax_out));
    CUDA_CHECK(cudaFree(per_sample_loss));
    CUDA_CHECK(cudaFree(d_labels));
    free(per_sample_loss_cpu);
}

// 工具函数：打印数组（调试用）
// Host-side 2D row<->col helpers (row-major <-> column-major)
void row_to_col_2d(const float* src, float* dst, int rows, int cols) {
    // src is row-major with shape (rows x cols)
    // dst will be column-major with same logical shape
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

void col_to_row_2d(const float* src, float* dst, int rows, int cols) {
    // src is column-major (cols blocks of rows), dst is row-major
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dst[r * cols + c] = src[c * rows + r];
        }
    }
}

// Wrapper: copy host row-major matrix to device as column-major
void cudaMemcpyHtoD_col_2d(float* dptr, const float* hptr, int rows, int cols) {
    size_t size = rows * cols * sizeof(float);
    float* tmp = (float*)malloc(size);
    if (!tmp) { printf("malloc failed\n"); exit(1); }
    row_to_col_2d(hptr, tmp, rows, cols);
    CUDA_CHECK(cudaMemcpy(dptr, tmp, size, cudaMemcpyHostToDevice));
    free(tmp);
}

// Wrapper: copy device column-major matrix to host row-major
void cudaMemcpyDtoH_col_2d(float* hptr, const float* dptr, int rows, int cols) {
    size_t size = rows * cols * sizeof(float);
    float* tmp = (float*)malloc(size);
    if (!tmp) { printf("malloc failed\n"); exit(1); }
    CUDA_CHECK(cudaMemcpy(tmp, dptr, size, cudaMemcpyDeviceToHost));
    col_to_row_2d(tmp, hptr, rows, cols);
    free(tmp);
}

// Forward declarations for FC kernels (defined later)
__global__ void fc_forward_kernel(const float* input, const float* weights, const float* bias, float* output,
                                  int batch_size, int in_features, int out_features);
__global__ void fc_grad_input_kernel(const float* grad_output, const float* weights, float* grad_input,
                                     int batch_size, int in_features, int out_features);
__global__ void fc_grad_weights_kernel(const float* grad_output, const float* input, float* grad_weights,
                                      int batch_size, int in_features, int out_features);
__global__ void fc_grad_bias_kernel(const float* grad_output, float* grad_bias, int batch_size, int out_features);

// ========================== 全连接层 CUDA 内核实现 ==========================
// fc_forward_kernel: for each output element (n, m), compute dot(input[n,:], weights[m,:]) + bias[m]
__global__ void fc_forward_kernel(const float* input, const float* weights, const float* bias, float* output,
                                  int batch_size, int in_features, int out_features) {
    int n = blockIdx.x * blockDim.x + threadIdx.x; // batch index
    int m = blockIdx.y * blockDim.y + threadIdx.y; // out feature index
    if (n >= batch_size || m >= out_features) return;

    const float* input_row = input + n * in_features;
    const float* weight_row = weights + m * in_features;
    float acc = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        acc += input_row[k] * weight_row[k];
    }
    float b = (bias != nullptr) ? bias[m] : 0.0f;
    output[n * out_features + m] = acc + b;
}

// fc_grad_input_kernel: grad_input[n,k] = sum_m grad_output[n,m] * weights[m,k]
__global__ void fc_grad_input_kernel(const float* grad_output, const float* weights, float* grad_input,
                                     int batch_size, int in_features, int out_features) {
    int n = blockIdx.x * blockDim.x + threadIdx.x; // batch
    int k = blockIdx.y * blockDim.y + threadIdx.y; // in feature
    if (n >= batch_size || k >= in_features) return;

    float acc = 0.0f;
    for (int m = 0; m < out_features; ++m) {
        acc += grad_output[n * out_features + m] * weights[m * in_features + k];
    }
    grad_input[n * in_features + k] = acc;
}

// fc_grad_weights_kernel: grad_weights[m,k] = sum_n grad_output[n,m] * input[n,k]
__global__ void fc_grad_weights_kernel(const float* grad_output, const float* input, float* grad_weights,
                                      int batch_size, int in_features, int out_features) {
    int m = blockIdx.x * blockDim.x + threadIdx.x; // out feature
    int k = blockIdx.y * blockDim.y + threadIdx.y; // in feature
    if (m >= out_features || k >= in_features) return;

    float acc = 0.0f;
    for (int n = 0; n < batch_size; ++n) {
        acc += grad_output[n * out_features + m] * input[n * in_features + k];
    }
    grad_weights[m * in_features + k] = acc;
}

// fc_grad_bias_kernel: grad_bias[m] = sum_n grad_output[n,m]
__global__ void fc_grad_bias_kernel(const float* grad_output, float* grad_bias, int batch_size, int out_features) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= out_features) return;
    float acc = 0.0f;
    for (int n = 0; n < batch_size; ++n) {
        acc += grad_output[n * out_features + m];
    }
    grad_bias[m] = acc;
}

// ========================== Dropout 层 ==========================

// Kernel: 根据 mask 决定是否丢弃
// 注意：训练时为了保持期望不变，需要除以 (1-prob)
__global__ void dropout_forward_kernel(const float* input, float* output, const float* mask, int size, float scale, float prob) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // mask 数组里存的是 0~1 的随机数
    // 如果 mask[idx] < prob，则丢弃(置0)；否则保留并缩放
    output[idx] = (mask[idx] < prob) ? 0.0f : input[idx] * scale;
}

__global__ void dropout_backward_kernel(const float* grad_output, float* grad_input, const float* mask, int size, float scale, float prob) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // 反向传播时，如果在前向中被丢弃了（mask < prob），则梯度也为0
    grad_input[idx] = (mask[idx] < prob) ? 0.0f : grad_output[idx] * scale;
}

void forward_dropout(float* input, float* output, float* mask, int size, float prob) {
    // 1. 生成随机 mask (0.0 ~ 1.0)
    // 我们复用 curand_generator 生成均匀分布随机数
    // 注意：这里直接把随机数写进 mask 指针指向的显存
    CURAND_CHECK(curandGenerateUniform(curand_generator, mask, size));
    
    float scale = 1.0f / (1.0f - prob);
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    
    dropout_forward_kernel<<<grid, block>>>(input, output, mask, size, scale, prob);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void backward_dropout(float* grad_output, float* grad_input, float* mask, int size, float prob) {
    float scale = 1.0f / (1.0f - prob);
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    
    dropout_backward_kernel<<<grid, block>>>(grad_output, grad_input, mask, size, scale, prob);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ========================== Batch Normalization Kernels ==========================

// Kernel 1: 计算均值和方差 (每个 Block 处理一个 Channel)
__global__ void bn_stats_kernel(const float* input, float* mean, float* var, int N, int C, int H, int W) {
    int c = blockIdx.x; // 当前处理的通道
    if (c >= C) return;

    // 每个 Block 的线程合作计算该通道所有像素的和
    // 这里简化处理：让 Thread 0 负责循环遍历 N*H*W (避免复杂的 reduction 代码，保证作业正确性)
    // 实际生产中会使用 Shared Memory 归约优化
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        int num_elements = N * H * W;
        
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = n * C * H * W + c * H * W + h * W + w;
                    float val = input[idx];
                    sum += val;
                    sq_sum += val * val;
                }
            }
        }
        
        float mu = sum / num_elements;
        mean[c] = mu;
        var[c] = (sq_sum / num_elements) - (mu * mu);
    }
}

// Kernel 2: 应用 BN 变换 (Normalization + Scale + Shift)
__global__ void bn_apply_kernel(const float* input, float* output, 
                                const float* mean, const float* var, 
                                const float* gamma, const float* bias, 
                                int N, int C, int H, int W, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    // 计算当前 idx 对应的 c
    int spatial = H * W;
    int c = (idx / spatial) % C;

    float mu = mean[c];
    float v = var[c];
    float std = sqrtf(v + epsilon);
    
    float x_hat = (input[idx] - mu) / std;
    output[idx] = gamma[c] * x_hat + bias[c];
}

// Kernel 3: 更新 Running Stats (Thread 0 处理)
__global__ void bn_update_running_kernel(float* running_mean, float* running_var, 
                                         const float* current_mean, const float* current_var, 
                                         int C, float momentum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C) return;
    
    running_mean[idx] = (1.0f - momentum) * running_mean[idx] + momentum * current_mean[idx];
    running_var[idx]  = (1.0f - momentum) * running_var[idx]  + momentum * current_var[idx];
}

// Kernel 4: BN 反向传播
// 这是一个融合 Kernel，直接计算 dx, dgamma, dbias
// 为了简化，我们分步做。首先计算 dgamma 和 dbias (类似于 stats kernel)
__global__ void bn_backward_param_kernel(const float* grad_output, const float* input, 
                                         const float* mean, const float* var, 
                                         float* grad_gamma, float* grad_bias, 
                                         int N, int C, int H, int W, float epsilon) {
    int c = blockIdx.x;
    if (c >= C) return;

    if (threadIdx.x == 0) {
        float d_gamma_sum = 0.0f;
        float d_bias_sum = 0.0f;
        float mu = mean[c];
        float v = var[c];
        float inv_std = 1.0f / sqrtf(v + epsilon);

        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = n * C * H * W + c * H * W + h * W + w;
                    float dout = grad_output[idx];
                    d_bias_sum += dout;
                    float x_hat = (input[idx] - mu) * inv_std;
                    d_gamma_sum += dout * x_hat;
                }
            }
        }
        grad_gamma[c] = d_gamma_sum;
        grad_bias[c]  = d_bias_sum;
    }
}

// Kernel 5: BN 反向传播计算 dx
// 这是一个复杂的公式，参见 BN 论文
__global__ void bn_backward_input_kernel(const float* grad_output, const float* input, float* grad_input,
                                         const float* mean, const float* var, const float* gamma,
                                         const float* grad_gamma, const float* grad_bias,
                                         int N, int C, int H, int W, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;

    int spatial = H * W;
    int c = (idx / spatial) % C;
    int M = N * H * W; // batch size equivalent

    float mu = mean[c];
    float v = var[c];
    float inv_std = 1.0f / sqrtf(v + epsilon);
    float x_hat = (input[idx] - mu) * inv_std;
    float g = gamma[c];

    // 标准 BN 反向公式
    // dx = (1/M) * (gamma/std) * (M*dy - sum(dy) - x_hat*sum(dy*x_hat))
    // 其中 sum(dy) 就是 grad_bias, sum(dy*x_hat) 就是 grad_gamma
    
    float dy = grad_output[idx];
    float d_beta = grad_bias[c];
    float d_gamma = grad_gamma[c];

    grad_input[idx] = (g * inv_std / M) * (M * dy - d_beta - x_hat * d_gamma);
}


// --- Host Functions ---

void forward_bn_train(float* input, float* output, float* save_mean, float* save_var, 
                      float* running_mean, float* running_var, 
                      float* gamma, float* bias, 
                      int N, int C, int H, int W, float momentum, float epsilon) {
    // 1. Calculate Mean & Var
    bn_stats_kernel<<<C, 1>>>(input, save_mean, save_var, N, C, H, W);
    CUDA_CHECK(cudaGetLastError());

    // 2. Update Running Stats
    dim3 block_stat(256);
    dim3 grid_stat((C + block_stat.x - 1) / block_stat.x);
    bn_update_running_kernel<<<grid_stat, block_stat>>>(running_mean, running_var, save_mean, save_var, C, momentum);

    // 3. Apply BN
    int total = N * C * H * W;
    dim3 block_apply(256);
    dim3 grid_apply((total + block_apply.x - 1) / block_apply.x);
    bn_apply_kernel<<<grid_apply, block_apply>>>(input, output, save_mean, save_var, gamma, bias, N, C, H, W, epsilon);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void forward_bn_test(float* input, float* output, 
                     float* running_mean, float* running_var, 
                     float* gamma, float* bias, 
                     int N, int C, int H, int W, float epsilon) {
    // Test mode: simply apply using running stats
    int total = N * C * H * W;
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);
    bn_apply_kernel<<<grid, block>>>(input, output, running_mean, running_var, gamma, bias, N, C, H, W, epsilon);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void backward_bn(float* grad_output, float* input, float* grad_input, 
                 float* grad_gamma, float* grad_bias, 
                 float* save_mean, float* save_var, float* gamma, 
                 int N, int C, int H, int W, float epsilon) {
    // 1. Calculate Gradients for Gamma and Bias
    bn_backward_param_kernel<<<C, 1>>>(grad_output, input, save_mean, save_var, grad_gamma, grad_bias, N, C, H, W, epsilon);
    CUDA_CHECK(cudaGetLastError());

    // 2. Calculate Gradient for Input
    int total = N * C * H * W;
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);
    bn_backward_input_kernel<<<grid, block>>>(grad_output, input, grad_input, save_mean, save_var, gamma, grad_gamma, grad_bias, N, C, H, W, epsilon);
    CUDA_CHECK(cudaDeviceSynchronize());
}
