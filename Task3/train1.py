# import numpy as np
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from autograd import Tensor, SGD
# import cuda_net

# # 1. 初始化环境
# cuda_net.init_context()

# # 2. 数据准备 (使用 torchvision)
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# # 3. 参数初始化 (手动创建网络参数)
# # 定义网络结构：Conv(3, 16, 3x3) -> MaxPool -> FC(16*16*16, 10)
# w1 = Tensor(np.random.randn(16, 3, 3, 3) * 0.01, requires_grad=True)
# b1 = Tensor(np.zeros(16), requires_grad=True)
# w2 = Tensor(np.random.randn(10, 16*16*16) * 0.01, requires_grad=True) # 假设池化后是16x16
# b2 = Tensor(np.zeros(10), requires_grad=True)

# params = [w1, b1, w2, b2]
# optimizer = SGD(params, lr=0.001)

# # 4. 训练循环
# for epoch in range(5):
#     running_loss = 0.0
#     for i, (inputs, labels) in enumerate(trainloader):
#         N, C, H, W = inputs.shape
#         labels_np = labels.numpy().astype(np.int32)
        
#         # --- 前向传播 ---
#         # Conv
#         out_conv_ptr = cuda_net.alloc_gpu(N * 16 * 32 * 32)
#         cuda_net.forward_conv(cuda_net.to_gpu(inputs.numpy()), out_conv_ptr, w1.ptr, b1.ptr, N, 3, 32, 32, 16)
        
#         # ReLU
#         out_relu_ptr = cuda_net.alloc_gpu(N * 16 * 32 * 32)
#         cuda_net.forward_relu(out_conv_ptr, out_relu_ptr, N * 16 * 32 * 32)
        
#         # Pool
#         mask_ptr = cuda_net.alloc_gpu(N * 16 * 32 * 32)
#         out_pool_ptr = cuda_net.alloc_gpu(N * 16 * 16 * 16)
#         cuda_net.forward_pool(out_relu_ptr, out_pool_ptr, mask_ptr, N, 16, 32, 32)
        
#         # FC
#         out_fc_ptr = cuda_net.alloc_gpu(N * 10)
#         cuda_net.forward_fc(out_pool_ptr, out_fc_ptr, w2.ptr, b2.ptr, N, 16*16*16, 10)
        
#         # Loss & Softmax
#         grad_fc_ptr = cuda_net.alloc_gpu(N * 10)
#         loss = cuda_net.cross_entropy(out_fc_ptr, labels_np, grad_fc_ptr, N, 10)
        
#         # --- 反向传播 (逆序手动调用) ---
#         # 1. FC Backward
#         gi_fc = cuda_net.alloc_gpu(N * 16*16*16)
#         cuda_net.backward_fc(out_pool_ptr, out_fc_ptr, w2.ptr, b2.ptr, N, 16*16*16, 10, grad_fc_ptr, gi_fc, w2.grad_ptr, b2.grad_ptr)
        
#         # 2. Pool Backward
#         gi_pool = cuda_net.alloc_gpu(N * 16 * 32 * 32)
#         cuda_net.backward_pool(gi_fc, mask_ptr, gi_pool, N, 16, 32, 32)
        
#         # 3. ReLU Backward
#         gi_relu = cuda_net.alloc_gpu(N * 16 * 32 * 32)
#         cuda_net.backward_relu(out_conv_ptr, gi_pool, gi_relu, N * 16 * 32 * 32)
        
#         # 4. Conv Backward
#         dummy_gi = cuda_net.alloc_gpu(N * 3 * 32 * 32)
#         cuda_net.backward_conv(cuda_net.to_gpu(inputs.numpy()), w1.ptr, gi_relu, dummy_gi, w1.grad_ptr, b1.grad_ptr, N, 3, 32, 32, 16)
        
#         # --- 更新 ---
#         optimizer.step()
        
#         # 释放中间变量内存 (非常重要，否则显存会炸)
#         for p in [out_conv_ptr, out_relu_ptr, out_pool_ptr, out_fc_ptr, mask_ptr, grad_fc_ptr, gi_fc, gi_pool, gi_relu, dummy_gi]:
#             cuda_net.free_gpu(p)

#         running_loss += loss
#         if i % 100 == 99:
#             print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
#             running_loss = 0.0

# cuda_net.destroy_context()
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from autograd import Tensor, SGD
import cuda_net 

# =================配置参数=================
BATCH_SIZE = 64
LR = 0.001 
EPOCHS = 60
DATA_ROOT = './data'

# =================数据准备=================
print("正在加载数据...")
# CIFAR-10 标准预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # CIFAR10 均值方差
])

trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# =================模型初始化=================
# 假设结构: Conv(3->16, 3x3) -> ReLU -> MaxPool(2x2) -> Flatten -> FC(16*16*16 -> 10)
# 图片 32x32 -> Conv(pad=1) -> 32x32 -> Pool -> 16x16
print("正在初始化模型参数...")
cuda_net.init_context()

# 权重初始化 (Xavier/Kaiming 简单模拟)
w1_np = np.random.randn(16, 3, 3, 3).astype(np.float32) * np.sqrt(2. / (3*3*3))
b1_np = np.zeros(16, dtype=np.float32)
w2_np = np.random.randn(10, 16 * 16 * 16).astype(np.float32) * np.sqrt(2. / (16*16*16))
b2_np = np.zeros(10, dtype=np.float32)

w1 = Tensor(w1_np, requires_grad=True)
b1 = Tensor(b1_np, requires_grad=True)
w2 = Tensor(w2_np, requires_grad=True)
b2 = Tensor(b2_np, requires_grad=True)

optimizer = SGD([w1, b1, w2, b2], lr=LR)

# =================辅助函数=================

def inference(inputs_np, labels_np):
    """
    只做前向传播，计算Loss和Accuracy，并立即释放显存。
    用于验证集/测试集评估。
    """
    N, C, H, W = inputs_np.shape
    
    # 1. Conv
    d_in = cuda_net.to_gpu(inputs_np)
    d_conv = cuda_net.alloc_gpu(N * 16 * 32 * 32)
    cuda_net.forward_conv(d_in, d_conv, w1.ptr, b1.ptr, N, 3, 32, 32, 16)
    cuda_net.free_gpu(d_in) # 释放输入
    
    # 2. ReLU
    d_relu = cuda_net.alloc_gpu(N * 16 * 32 * 32)
    cuda_net.forward_relu(d_conv, d_relu, N * 16 * 32 * 32)
    cuda_net.free_gpu(d_conv) # 释放conv结果
    
    # 3. Pool
    d_pool = cuda_net.alloc_gpu(N * 16 * 16 * 16)
    d_mask = cuda_net.alloc_gpu(N * 16 * 32 * 32) # Inference其实不需要mask，但为了复用接口
    cuda_net.forward_pool(d_relu, d_pool, d_mask, N, 16, 32, 32)
    cuda_net.free_gpu(d_relu)
    cuda_net.free_gpu(d_mask) # 这里的mask不需要用于反向，直接释放
    
    # 4. FC
    d_fc = cuda_net.alloc_gpu(N * 10)
    cuda_net.forward_fc(d_pool, d_fc, w2.ptr, b2.ptr, N, 16*16*16, 10)
    cuda_net.free_gpu(d_pool)
    
    # 5. Loss & Acc 计算
    # 注意：cross_entropy 内部通常不释放 logits，我们需要手动释放
    d_grad_dummy = cuda_net.alloc_gpu(N * 10) # 只是为了调用接口，实际上不需要梯度
    loss = cuda_net.cross_entropy(d_fc, labels_np, d_grad_dummy, N, 10)
    
    # 拷贝 logits 回 CPU 计算准确率
    logits = cuda_net.to_cpu(d_fc, [N, 10])
    preds = np.argmax(logits, axis=1)
    correct = np.sum(preds == labels_np)
    
    # 清理最后剩下的
    cuda_net.free_gpu(d_fc)
    cuda_net.free_gpu(d_grad_dummy)
    
    return loss, correct

def evaluate(dataloader):
    """遍历数据集计算平均Loss和总准确率"""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs_np = inputs.numpy()
        labels_np = labels.numpy().astype(np.int32)
        
        loss, correct = inference(inputs_np, labels_np)
        
        total_loss += loss * inputs.shape[0] # 因为 loss 通常是 mean
        total_correct += correct
        total_samples += inputs.shape[0]
        
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc

# =================训练循环=================
history = {
    'train_loss': [], 'train_acc': [],
    'test_loss': [], 'test_acc': []
}

print(f"开始训练，共 {EPOCHS} 个 Epoch...")
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    train_loss_sum = 0.0
    train_correct_sum = 0
    train_samples = 0
    
    # --- Training Step ---
    for i, (inputs, labels) in enumerate(trainloader):
        N, C, H, W = inputs.shape
        inputs_np = inputs.numpy()
        labels_np = labels.numpy().astype(np.int32)
        
        # 1. 前向传播 (必须保留指针用于反向)
        d_in = cuda_net.to_gpu(inputs_np)
        
        d_conv = cuda_net.alloc_gpu(N * 16 * 32 * 32)
        cuda_net.forward_conv(d_in, d_conv, w1.ptr, b1.ptr, N, 3, 32, 32, 16)
        
        d_relu = cuda_net.alloc_gpu(N * 16 * 32 * 32)
        cuda_net.forward_relu(d_conv, d_relu, N * 16 * 32 * 32)
        
        d_pool = cuda_net.alloc_gpu(N * 16 * 16 * 16)
        d_mask = cuda_net.alloc_gpu(N * 16 * 32 * 32)
        cuda_net.forward_pool(d_relu, d_pool, d_mask, N, 16, 32, 32)
        
        d_fc = cuda_net.alloc_gpu(N * 10)
        cuda_net.forward_fc(d_pool, d_fc, w2.ptr, b2.ptr, N, 16*16*16, 10)
        
        d_grad_fc = cuda_net.alloc_gpu(N * 10)
        loss = cuda_net.cross_entropy(d_fc, labels_np, d_grad_fc, N, 10)
        
        # 2. 统计 Train Accuracy (Batch)
        logits = cuda_net.to_cpu(d_fc, [N, 10])
        preds = np.argmax(logits, axis=1)
        train_correct_sum += np.sum(preds == labels_np)
        train_loss_sum += loss * N
        train_samples += N
        
        # 3. 反向传播
        d_gi_fc = cuda_net.alloc_gpu(N * 16*16*16)
        cuda_net.backward_fc(d_pool, d_fc, w2.ptr, b2.ptr, N, 16*16*16, 10, d_grad_fc, d_gi_fc, w2.grad_ptr, b2.grad_ptr)
        
        d_gi_pool = cuda_net.alloc_gpu(N * 16 * 32 * 32)
        cuda_net.backward_pool(d_gi_fc, d_mask, d_gi_pool, N, 16, 32, 32)
        
        d_gi_relu = cuda_net.alloc_gpu(N * 16 * 32 * 32)
        cuda_net.backward_relu(d_conv, d_gi_pool, d_gi_relu, N * 16 * 32 * 32)
        
        d_gi_conv_dummy = cuda_net.alloc_gpu(N * 3 * 32 * 32)
        cuda_net.backward_conv(d_in, w1.ptr, d_gi_relu, d_gi_conv_dummy, w1.grad_ptr, b1.grad_ptr, N, 3, 32, 32, 16)
        
        # 4. 参数更新
        optimizer.step()
        
        # 5. 显存释放 (Train Loop 最关键的一步)
        ptrs_to_free = [d_in, d_conv, d_relu, d_pool, d_mask, d_fc, d_grad_fc, 
                        d_gi_fc, d_gi_pool, d_gi_relu, d_gi_conv_dummy]
        for p in ptrs_to_free:
            cuda_net.free_gpu(p)
            
        if i % 100 == 0:
            print(f"  [Batch {i}/{len(trainloader)}] Loss: {loss:.4f}")

    # --- Epoch End Statistics ---
    train_avg_loss = train_loss_sum / train_samples
    train_acc = train_correct_sum / train_samples
    
    # --- Validation/Test Step ---
    print("  正在评估测试集...")
    test_loss, test_acc = evaluate(testloader)
    
    epoch_time = time.time() - epoch_start
    
    # 记录历史
    history['train_loss'].append(train_avg_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Time: {epoch_time:.1f}s | "
          f"Train Loss: {train_avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

total_time = time.time() - start_time
print(f"\n训练结束！总耗时: {total_time/60:.2f} 分钟")

# =================清理资源=================
cuda_net.destroy_context()

# =================绘图功能=================
plt.figure(figsize=(12, 5))

# 1. Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['test_loss'], label='Test Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)

# 2. Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['test_acc'], label='Test Acc', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_result.png') # 保存图片
print("结果曲线已保存为 training_result.png")
plt.show()