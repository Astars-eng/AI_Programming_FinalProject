#ifndef CNN_MODULES_H  // 防止重复包含（必加）
#define CNN_MODULES_H
// 若不是用 nvcc 编译（例如用 g++ 编译 pybind 模块），将 __global__ 定义为空以便编译
#ifndef __CUDACC__
#define __global__
#endif

__global__ void relu_kernel(const float* d_input, float* d_output, int size);
__global__ void relu_backward_kernel(const float* d_input, const float* d_grad_output, float* d_grad_input, int size);
__global__ void sigmoid_forward_kernel(const float* d_input, float* d_output, int size);
__global__ void sigmoid_backward_kernel(const float* d_output, const float* d_grad_output, float* d_grad_input, int size);
__global__ void flatten_kernel(
    const float* weights, 
    float* weights_flat, 
    int C_out, 
    int C_in, 
    int kernel_size, 
    int col_height
);
__global__ void reshape_kernel(
    const float* conv_out, 
    float* output, 
    int C_out, 
    int N, 
    int H, 
    int W, 
    int col_width
);
__global__ void flat_grad_kernel(
    const float* grad_output, 
    float* grad_output_flat, 
    int C_out, 
    int N, 
    int H, 
    int W, 
    int col_width
);
__global__ void reshape_weight_kernel(
    const float* grad_weights_flat, 
    float* grad_weights, 
    int C_out, 
    int C_in, 
    int kernel_size, 
    int col_height
);
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
);
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
);
__global__ void softmax_max_kernel(
    const float* input, 
    float* row_max, 
    int batch_size, 
    int features
);
__global__ void softmax_sum_kernel(
    const float* output, 
    float* row_sum, 
    int batch_size, 
    int features
);
__global__ void ce_loss_kernel(
    const float* softmax_out, 
    const int* labels, 
    float* per_sample_loss, 
    int batch_size, 
    int features
);
__global__ void ce_grad_kernel(
    const float* softmax_out, 
    const int* labels, 
    float* grad_logits, 
    float scale, 
    int batch_size, 
    int features
);
void forward_relu_gpu(const float* d_input, float* d_output, int size);
void backward_relu_gpu(const float* d_input, const float* d_grad_output, float* d_grad_input, int size);
void forward_sigmoid_gpu(const float* d_input, float* d_output, int size);
void backward_sigmoid_gpu(const float* d_output, const float* d_grad_output, float* d_grad_input, int size);
void init_cublas();
void destroy_cublas();
void create_ones_gpu(float** ones,int rows,int cols);
void init_weights_gpu(float** weights, int rows, int cols);
void init_bias_gpu(float** bias, int size);
void forward_fc(float* input, float* output, float* weights, float* bias,
                int batch_size, int in_features, int out_features);
void backward_fc(float* input, float* output, float* weights, float* bias,
                 int batch_size, int in_features, int out_features,
                 float* grad_output, float* grad_input, float* grad_weights,
                 float* grad_bias);
__global__ void im2col_kernel(const float* input, float* col,
                              int N, int C_in, int H, int W, int kernel_size, int padding);
void im2col(float* input, float* col, int N, int C_in, int H, int W, int kernel_size, int padding);
__global__ void col2im_kernel(const float* col, float* output,
                              int N, int C_in, int H, int W, int kernel_size, int padding);
void col2im(float* col, float* output, int N, int C_in, int H, int W, int kernel_size, int padding);
void forward_conv(float* input, float* output, float* weights, float* bias,
                  int N, int C_in, int H, int W, int C_out);
void backward_conv(float* input, float* weights, float* grad_output,
                   float* grad_input, float* grad_weights, float* grad_bias,
                   int N, int C_in, int H, int W, int C_out);
void forward_max_pool(float* input, float* output, float* mask,
                      int N, int C, int H, int W);
void backward_max_pool(float* grad_output, float* mask, float* grad_input,
                       int N, int C, int H, int W);
void stable_softmax(float* input, float* output, int batch_size, int features);
void cross_entropy_softmax(float* logits, int* labels, float* loss, float* grad_logits,
                           int batch_size, int features);
void row_to_col_2d(const float* src, float* dst, int rows, int cols);
void col_to_row_2d(const float* src, float* dst, int rows, int cols);
void cudaMemcpyHtoD_col_2d(float* dptr, const float* hptr, int rows, int cols);
void cudaMemcpyDtoH_col_2d(float* hptr, const float* dptr, int rows, int cols);

// Dropout 相关
void forward_dropout(float* input, float* output, float* mask, int size, float prob);
void backward_dropout(float* grad_output, float* grad_input, float* mask, int size, float prob);

// ========================== Batch Normalization ==========================
// 训练时前向：计算并保存 mean/var，更新 running stats
void forward_bn_train(float* input, float* output, float* save_mean, float* save_var, 
                      float* running_mean, float* running_var, 
                      float* gamma, float* bias, 
                      int N, int C, int H, int W, float momentum, float epsilon);

// 测试时前向：直接使用 running stats
void forward_bn_test(float* input, float* output, 
                     float* running_mean, float* running_var, 
                     float* gamma, float* bias, 
                     int N, int C, int H, int W, float epsilon);

// 反向传播
void backward_bn(float* grad_output, float* input, float* grad_input, 
                 float* grad_gamma, float* grad_bias, 
                 float* save_mean, float* save_var, float* gamma, 
                 int N, int C, int H, int W, float epsilon);

#endif // CNN_MODULES_H