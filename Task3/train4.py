import time
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 引入你的 CUDA 模块
import cuda_net

# ================= 配置参数 =================
BATCH_SIZE = 64
EPOCHS = 100           # 30轮足够观察到BN带来的提升
LR_MAX = 0.02         # 初始学习率
LR_MIN = 0.0001       # 最低学习率
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4   # L2 正则化
DROPOUT_RATE = 0.5    # 全连接层 Dropout
BN_MOMENTUM = 0.1     # BN 统计量动量
BN_EPS = 1e-5         # 防止除零

# ================= 辅助类 (Tensor & SGD) =================
class Tensor:
    def __init__(self, data, shape=None, requires_grad=False):
        if isinstance(data, np.ndarray):
            self.ptr = cuda_net.to_gpu(data.astype(np.float32))
            self.shape = data.shape
        else:
            self.ptr = data 
            self.shape = shape
        
        self.requires_grad = requires_grad
        self.grad_ptr = None
        if requires_grad:
            size = int(np.prod(self.shape))
            self.grad_ptr = cuda_net.alloc_gpu(size)

class SGD:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros(p.shape, dtype=np.float32) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad_ptr:
                # 1. GPU -> CPU
                grad = cuda_net.to_cpu(p.grad_ptr, list(p.shape))
                p_data = cuda_net.to_cpu(p.ptr, list(p.shape))
                
                # 2. Weight Decay (L2 Regularization)
                if self.weight_decay > 0:
                    grad += self.weight_decay * p_data
                
                # 3. Momentum Update
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                
                # 4. Apply Update
                p_data -= self.lr * self.velocities[i]
                
                # 5. CPU -> GPU (Re-alloc to be safe)
                cuda_net.free_gpu(p.ptr)
                p.ptr = cuda_net.to_gpu(p_data)

    def zero_grad(self):
        # 简单起见，这里不需要显式清零，因为我们在 backward 前都是 alloc 新的 grad 显存
        # 实际框架中需要 memset 0
        pass

# ================= 数据准备 (含增强) =================
print("正在加载数据...")

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ================= 模型初始化 =================
cuda_net.init_context()

def kaiming_init(shape):
    fan_in = np.prod(shape[1:])
    return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)

print("正在初始化参数 (Conv-BN-ReLU-Pool x2 -> FC-Drop-FC)...")

# --- Block 1: 3 -> 32 ---
w1 = Tensor(kaiming_init((32, 3, 3, 3)), requires_grad=True)
b1 = Tensor(np.zeros(32), requires_grad=True)
# BN1 Params
gamma1 = Tensor(np.ones(32, dtype=np.float32), requires_grad=True)
beta1  = Tensor(np.zeros(32, dtype=np.float32), requires_grad=True)
running_mean1 = cuda_net.alloc_gpu(32) # Init 0
running_var1  = cuda_net.to_gpu(np.ones(32, dtype=np.float32)) # Init 1

# --- Block 2: 32 -> 64 ---
w2 = Tensor(kaiming_init((64, 32, 3, 3)), requires_grad=True)
b2 = Tensor(np.zeros(64), requires_grad=True)
# BN2 Params
gamma2 = Tensor(np.ones(64, dtype=np.float32), requires_grad=True)
beta2  = Tensor(np.zeros(64, dtype=np.float32), requires_grad=True)
running_mean2 = cuda_net.alloc_gpu(64)
running_var2  = cuda_net.to_gpu(np.ones(64, dtype=np.float32))

# --- FC 1: 64*8*8 -> 256 ---
w3 = Tensor(kaiming_init((256, 64 * 8 * 8)), requires_grad=True)
b3 = Tensor(np.zeros(256), requires_grad=True)

# --- FC 2: 256 -> 10 ---
w4 = Tensor(kaiming_init((10, 256)), requires_grad=True)
b4 = Tensor(np.zeros(10), requires_grad=True)

# 优化器管理所有 Trainable 参数 (Running stats 不在其中)
params = [w1, b1, gamma1, beta1, w2, b2, gamma2, beta2, w3, b3, w4, b4]
optimizer = SGD(params, lr=LR_MAX, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# ================= 前向传播逻辑 =================
def forward_pass(inputs_ptr, N, training=True):
    # -------- Block 1 --------
    # 1. Conv
    c1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
    cuda_net.forward_conv(inputs_ptr, c1, w1.ptr, b1.ptr, N, 3, 32, 32, 32)
    
    # 2. BN
    bn1_out = cuda_net.alloc_gpu(N * 32 * 32 * 32)
    # 训练时需要保存 mean/var 给反向用
    save_mean1 = cuda_net.alloc_gpu(32) 
    save_var1  = cuda_net.alloc_gpu(32)
    
    if training:
        cuda_net.forward_bn_train(c1, bn1_out, save_mean1, save_var1, running_mean1, running_var1, 
                                  gamma1.ptr, beta1.ptr, N, 32, 32, 32, BN_MOMENTUM, BN_EPS)
    else:
        # 测试时不用 save_mean/var，但为了保持返回 tuple 结构一致，我们分配个空的或者处理为None
        # 这里分配最小内存占位即可
        cuda_net.forward_bn_test(c1, bn1_out, running_mean1, running_var1, 
                                 gamma1.ptr, beta1.ptr, N, 32, 32, 32, BN_EPS)

    # 3. ReLU
    r1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
    cuda_net.forward_relu(bn1_out, r1, N * 32 * 32 * 32)
    
    # 4. Pool
    p1 = cuda_net.alloc_gpu(N * 32 * 16 * 16)
    m1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
    cuda_net.forward_pool(r1, p1, m1, N, 32, 32, 32)
    
    # -------- Block 2 --------
    # 1. Conv
    c2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
    cuda_net.forward_conv(p1, c2, w2.ptr, b2.ptr, N, 32, 16, 16, 64)
    
    # 2. BN
    bn2_out = cuda_net.alloc_gpu(N * 64 * 16 * 16)
    save_mean2 = cuda_net.alloc_gpu(64)
    save_var2  = cuda_net.alloc_gpu(64)
    
    if training:
        cuda_net.forward_bn_train(c2, bn2_out, save_mean2, save_var2, running_mean2, running_var2,
                                  gamma2.ptr, beta2.ptr, N, 64, 16, 16, BN_MOMENTUM, BN_EPS)
    else:
        cuda_net.forward_bn_test(c2, bn2_out, running_mean2, running_var2,
                                 gamma2.ptr, beta2.ptr, N, 64, 16, 16, BN_EPS)
    
    # 3. ReLU
    r2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
    cuda_net.forward_relu(bn2_out, r2, N * 64 * 16 * 16)
    
    # 4. Pool
    p2 = cuda_net.alloc_gpu(N * 64 * 8 * 8)
    m2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
    cuda_net.forward_pool(r2, p2, m2, N, 64, 16, 16)
    
    # -------- FC Block --------
    # 1. FC1
    fc1 = cuda_net.alloc_gpu(N * 256)
    cuda_net.forward_fc(p2, fc1, w3.ptr, b3.ptr, N, 64*8*8, 256)
    
    # 2. ReLU
    r3 = cuda_net.alloc_gpu(N * 256)
    cuda_net.forward_relu(fc1, r3, N * 256)
    
    # 3. Dropout
    d_out = cuda_net.alloc_gpu(N * 256)
    d_mask = cuda_net.alloc_gpu(N * 256)
    if training:
        cuda_net.forward_dropout(r3, d_out, d_mask, N * 256, DROPOUT_RATE)
    else:
        # 传入 prob=0.0 等同于直通
        cuda_net.forward_dropout(r3, d_out, d_mask, N * 256, 0.0)
        
    # 4. FC2 (Logits)
    out = cuda_net.alloc_gpu(N * 10)
    cuda_net.forward_fc(d_out, out, w4.ptr, b4.ptr, N, 256, 10)
    
    # Cache all pointers needed for backward or free
    cache = (c1, bn1_out, save_mean1, save_var1, r1, p1, m1,
             c2, bn2_out, save_mean2, save_var2, r2, p2, m2,
             fc1, r3, d_out, d_mask)
    return out, cache

# ================= 评估函数 =================
def evaluate(loader):
    correct = 0
    total = 0
    
    # 切换为测试模式 (BN 使用 running stats, Dropout 关闭)
    for inputs, labels in loader:
        N = inputs.shape[0]
        d_in = cuda_net.to_gpu(inputs.numpy())
        labels_np = labels.numpy().astype(np.int32)
        
        logits_ptr, cache = forward_pass(d_in, N, training=False)
        
        # 拷贝结果回 CPU
        logits = cuda_net.to_cpu(logits_ptr, [N, 10])
        preds = np.argmax(logits, axis=1)
        correct += np.sum(preds == labels_np)
        total += N
        
        # 清理内存
        cuda_net.free_gpu(d_in)
        cuda_net.free_gpu(logits_ptr)
        for ptr in cache: cuda_net.free_gpu(ptr)
        
    return correct / total

# ================= 训练循环 =================
history = {'loss': [], 'acc': []}
print("开始训练...")
start_time = time.time()

for epoch in range(EPOCHS):
    # --- Cosine LR Scheduler ---
    new_lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * epoch / EPOCHS))
    optimizer.lr = new_lr
    
    train_loss = 0.0
    batch_count = 0
    
    for i, (inputs, labels) in enumerate(trainloader):
        N = inputs.shape[0]
        inputs_np = inputs.numpy()
        labels_np = labels.numpy().astype(np.int32)
        
        # 1. Forward
        d_in = cuda_net.to_gpu(inputs_np)
        logits_ptr, cache = forward_pass(d_in, N, training=True)
        (c1, bn1_out, save_mean1, save_var1, r1, p1, m1,
         c2, bn2_out, save_mean2, save_var2, r2, p2, m2,
         fc1, r3, d_out, d_mask) = cache
        
        # 2. Loss
        d_grad_logits = cuda_net.alloc_gpu(N * 10)
        loss = cuda_net.cross_entropy(logits_ptr, labels_np, d_grad_logits, N, 10)
        
        # 3. Backward (Manual Chain Rule)
        # ---------------- FC Block Backward ----------------
        d_gi_drop = cuda_net.alloc_gpu(N * 256)
        cuda_net.backward_fc(d_out, logits_ptr, w4.ptr, b4.ptr, N, 256, 10, 
                             d_grad_logits, d_gi_drop, w4.grad_ptr, b4.grad_ptr)
        
        d_gi_r3 = cuda_net.alloc_gpu(N * 256)
        cuda_net.backward_dropout(d_gi_drop, d_gi_r3, d_mask, N * 256, DROPOUT_RATE)
        
        d_gi_fc1 = cuda_net.alloc_gpu(N * 256)
        cuda_net.backward_relu(fc1, d_gi_r3, d_gi_fc1, N * 256)
        
        d_gi_p2 = cuda_net.alloc_gpu(N * 64*8*8)
        cuda_net.backward_fc(p2, fc1, w3.ptr, b3.ptr, N, 64*8*8, 256, 
                             d_gi_fc1, d_gi_p2, w3.grad_ptr, b3.grad_ptr)
        
        # ---------------- Block 2 Backward ----------------
        d_gi_r2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
        cuda_net.backward_pool(d_gi_p2, m2, d_gi_r2, N, 64, 16, 16)
        
        d_gi_bn2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
        cuda_net.backward_relu(bn2_out, d_gi_r2, d_gi_bn2, N * 64 * 16 * 16)
        
        d_gi_c2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
        cuda_net.backward_bn(d_gi_bn2, c2, d_gi_c2, gamma2.grad_ptr, beta2.grad_ptr, 
                             save_mean2, save_var2, gamma2.ptr, N, 64, 16, 16, BN_EPS)
                             
        d_gi_p1 = cuda_net.alloc_gpu(N * 32 * 16 * 16)
        cuda_net.backward_conv(p1, w2.ptr, d_gi_c2, d_gi_p1, w2.grad_ptr, b2.grad_ptr, N, 32, 16, 16, 64)
        
        # ---------------- Block 1 Backward ----------------
        d_gi_r1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
        cuda_net.backward_pool(d_gi_p1, m1, d_gi_r1, N, 32, 32, 32)
        
        d_gi_bn1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
        cuda_net.backward_relu(bn1_out, d_gi_r1, d_gi_bn1, N * 32 * 32 * 32)
        
        d_gi_c1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
        cuda_net.backward_bn(d_gi_bn1, c1, d_gi_c1, gamma1.grad_ptr, beta1.grad_ptr, 
                             save_mean1, save_var1, gamma1.ptr, N, 32, 32, 32, BN_EPS)
        
        d_dummy = cuda_net.alloc_gpu(N * 3 * 32 * 32)
        cuda_net.backward_conv(d_in, w1.ptr, d_gi_c1, d_dummy, w1.grad_ptr, b1.grad_ptr, N, 3, 32, 32, 32)

        # 4. Update
        optimizer.step()
        
        # 5. Clean up (Strictly free ALL allocated pointers)
        ptrs_to_free = [
            d_in, d_grad_logits, logits_ptr,
            d_gi_drop, d_gi_r3, d_gi_fc1, d_gi_p2, 
            d_gi_r2, d_gi_bn2, d_gi_c2, d_gi_p1, 
            d_gi_r1, d_gi_bn1, d_gi_c1, d_dummy
        ]
        for p in ptrs_to_free: cuda_net.free_gpu(p)
        for p in cache: cuda_net.free_gpu(p) # Free Forward Cache
        
        train_loss += loss
        batch_count += 1
        
        if i % 100 == 0:
            print(f"  [Epoch {epoch+1}, Batch {i}] Loss: {loss:.4f}")

    # --- Epoch End ---
    avg_loss = train_loss / batch_count
    val_acc = evaluate(testloader)
    history['loss'].append(avg_loss)
    history['acc'].append(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Time: {time.time()-start_time:.1f}s | "
          f"LR: {new_lr:.5f} | Loss: {avg_loss:.4f} | Test Acc: {val_acc*100:.2f}%")

# ================= 绘图与保存 =================
cuda_net.destroy_context()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['acc'], label='Test Accuracy', color='orange')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.grid(True)

plt.savefig('final_result_with_bn.png')
plt.show()

print(f"最终准确率: {history['acc'][-1]*100:.2f}%")