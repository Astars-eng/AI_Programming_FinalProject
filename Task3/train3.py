import time
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 引入编译好的 CUDA 模块
import cuda_net

# ================= 配置参数 =================
BATCH_SIZE = 64
EPOCHS = 30           # 30轮，配合余弦退火
LR_MAX = 0.02         # 初始学习率
LR_MIN = 0.0001       # 最低学习率
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4 
DROPOUT_RATE = 0.5  

# ================= 辅助类 =================
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
                grad = cuda_net.to_cpu(p.grad_ptr, list(p.shape))
                p_data = cuda_net.to_cpu(p.ptr, list(p.shape))
                
                # Weight Decay
                if self.weight_decay > 0:
                    grad += self.weight_decay * p_data
                
                # Momentum
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                
                # Update
                p_data -= self.lr * self.velocities[i]
                
                cuda_net.free_gpu(p.ptr)
                p.ptr = cuda_net.to_gpu(p_data)

# ================= 数据加载 =================
print("正在加载数据...")

# 训练集：数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 测试集：仅归一化
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ================= 模型初始化 (VGG-Lite) =================
cuda_net.init_context()

def kaiming_init(shape):
    fan_in = np.prod(shape[1:])
    return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)

print("正在初始化参数...")

# Layer 1: Conv 3->32
w1 = Tensor(kaiming_init((32, 3, 3, 3)), requires_grad=True)
b1 = Tensor(np.zeros(32), requires_grad=True)

# Layer 2: Conv 32->64
w2 = Tensor(kaiming_init((64, 32, 3, 3)), requires_grad=True)
b2 = Tensor(np.zeros(64), requires_grad=True)

# Layer 3: FC 64*8*8 -> 256
w3 = Tensor(kaiming_init((256, 64 * 8 * 8)), requires_grad=True)
b3 = Tensor(np.zeros(256), requires_grad=True)

# Layer 4: FC 256 -> 10
w4 = Tensor(kaiming_init((10, 256)), requires_grad=True)
b4 = Tensor(np.zeros(10), requires_grad=True)

params = [w1, b1, w2, b2, w3, b3, w4, b4]
optimizer = SGD(params, lr=LR_MAX, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# ================= 前向传播 (含 Dropout) =================
def forward_pass(inputs_ptr, N, training=True):
    # Conv1 -> ReLU -> Pool1
    c1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
    cuda_net.forward_conv(inputs_ptr, c1, w1.ptr, b1.ptr, N, 3, 32, 32, 32)
    r1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
    cuda_net.forward_relu(c1, r1, N * 32 * 32 * 32)
    p1 = cuda_net.alloc_gpu(N * 32 * 16 * 16)
    m1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
    cuda_net.forward_pool(r1, p1, m1, N, 32, 32, 32)
    
    # Conv2 -> ReLU -> Pool2
    c2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
    cuda_net.forward_conv(p1, c2, w2.ptr, b2.ptr, N, 32, 16, 16, 64)
    r2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
    cuda_net.forward_relu(c2, r2, N * 64 * 16 * 16)
    p2 = cuda_net.alloc_gpu(N * 64 * 8 * 8)
    m2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
    cuda_net.forward_pool(r2, p2, m2, N, 64, 16, 16)
    
    # Flatten -> FC1 -> ReLU
    fc1 = cuda_net.alloc_gpu(N * 256)
    cuda_net.forward_fc(p2, fc1, w3.ptr, b3.ptr, N, 64*8*8, 256)
    r3 = cuda_net.alloc_gpu(N * 256)
    cuda_net.forward_relu(fc1, r3, N * 256)
    
    # --- Dropout ---
    d_out = cuda_net.alloc_gpu(N * 256)
    d_mask = cuda_net.alloc_gpu(N * 256)
    if training:
        # 训练时：随机丢弃，且 scale
        cuda_net.forward_dropout(r3, d_out, d_mask, N * 256, DROPOUT_RATE)
    else:
        # 测试时：直通 (prob=0.0)
        cuda_net.forward_dropout(r3, d_out, d_mask, N * 256, 0.0)

    # FC2 (Output)
    out = cuda_net.alloc_gpu(N * 10)
    cuda_net.forward_fc(d_out, out, w4.ptr, b4.ptr, N, 256, 10)

    # Cache
    cache = (c1, r1, p1, m1, c2, r2, p2, m2, fc1, r3, d_out, d_mask)
    return out, cache

# ================= 评估函数 =================
def evaluate(loader):
    """计算验证集的 Loss 和 Accuracy"""
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        N = inputs.shape[0]
        inputs_np = inputs.numpy()
        labels_np = labels.numpy().astype(np.int32)
        d_in = cuda_net.to_gpu(inputs_np)
        
        # Inference mode
        logits_ptr, cache = forward_pass(d_in, N, training=False)
        
        # 1. Calc Loss
        d_grad_dummy = cuda_net.alloc_gpu(N * 10)
        loss = cuda_net.cross_entropy(logits_ptr, labels_np, d_grad_dummy, N, 10)
        total_loss += loss * N
        
        # 2. Calc Acc (CPU)
        logits = cuda_net.to_cpu(logits_ptr, [N, 10])
        preds = np.argmax(logits, axis=1)
        correct += np.sum(preds == labels_np)
        total += N
        
        # 3. Clean up
        cuda_net.free_gpu(d_in)
        cuda_net.free_gpu(d_grad_dummy)
        cuda_net.free_gpu(logits_ptr)
        for ptr in cache: cuda_net.free_gpu(ptr)
        
    return total_loss / total, correct / total

# ================= 主循环 =================
history = {
    'train_loss': [], 'train_acc': [],
    'test_loss': [], 'test_acc': []
}

print(f"开始训练 (Dropout Rate: {DROPOUT_RATE}, Cosine LR)...")
start_time = time.time()

for epoch in range(EPOCHS):
    # --- LR Scheduler ---
    new_lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * epoch / EPOCHS))
    optimizer.lr = new_lr
    
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0
    
    for i, (inputs, labels) in enumerate(trainloader):
        N = inputs.shape[0]
        inputs_np = inputs.numpy()
        labels_np = labels.numpy().astype(np.int32)
        
        # 1. Forward
        d_in = cuda_net.to_gpu(inputs_np)
        logits_ptr, cache = forward_pass(d_in, N, training=True)
        (c1, r1, p1, m1, c2, r2, p2, m2, fc1, r3, d_out, d_mask) = cache
        
        # 2. Loss
        d_grad_logits = cuda_net.alloc_gpu(N * 10)
        loss = cuda_net.cross_entropy(logits_ptr, labels_np, d_grad_logits, N, 10)
        
        # --- 统计 Training Acc (新增) ---
        # 拷贝 logits 回 CPU 算准确率，虽然有点慢，但为了看曲线是值得的
        cpu_logits = cuda_net.to_cpu(logits_ptr, [N, 10])
        preds = np.argmax(cpu_logits, axis=1)
        train_correct += np.sum(preds == labels_np)
        train_total += N
        train_loss_sum += loss * N
        
        # 3. Backward
        d_gi_drop = cuda_net.alloc_gpu(N * 256)
        cuda_net.backward_fc(d_out, logits_ptr, w4.ptr, b4.ptr, N, 256, 10, d_grad_logits, d_gi_drop, w4.grad_ptr, b4.grad_ptr)
        
        d_gi_r3 = cuda_net.alloc_gpu(N * 256)
        cuda_net.backward_dropout(d_gi_drop, d_gi_r3, d_mask, N * 256, DROPOUT_RATE) # Dropout Back
        
        d_gi_fc1 = cuda_net.alloc_gpu(N * 256)
        cuda_net.backward_relu(fc1, d_gi_r3, d_gi_fc1, N * 256)
        
        d_gi_p2 = cuda_net.alloc_gpu(N * 64*8*8)
        cuda_net.backward_fc(p2, fc1, w3.ptr, b3.ptr, N, 64*8*8, 256, d_gi_fc1, d_gi_p2, w3.grad_ptr, b3.grad_ptr)
        
        d_gi_r2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
        cuda_net.backward_pool(d_gi_p2, m2, d_gi_r2, N, 64, 16, 16)
        
        d_gi_c2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
        cuda_net.backward_relu(c2, d_gi_r2, d_gi_c2, N * 64 * 16 * 16)
        
        d_gi_p1 = cuda_net.alloc_gpu(N * 32 * 16 * 16)
        cuda_net.backward_conv(p1, w2.ptr, d_gi_c2, d_gi_p1, w2.grad_ptr, b2.grad_ptr, N, 32, 16, 16, 64)
        
        d_gi_r1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
        cuda_net.backward_pool(d_gi_p1, m1, d_gi_r1, N, 32, 32, 32)
        
        d_gi_c1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
        cuda_net.backward_relu(c1, d_gi_r1, d_gi_c1, N * 32 * 32 * 32)
        
        d_dummy = cuda_net.alloc_gpu(N * 3 * 32 * 32)
        cuda_net.backward_conv(d_in, w1.ptr, d_gi_c1, d_dummy, w1.grad_ptr, b1.grad_ptr, N, 3, 32, 32, 32)
        
        # 4. Update
        optimizer.step()
        
        # 5. Free
        cuda_net.free_gpu(d_in)
        cuda_net.free_gpu(d_grad_logits)
        for p in cache: cuda_net.free_gpu(p)
        for p in [d_gi_drop, d_gi_r3, d_gi_fc1, d_gi_p2, d_gi_r2, d_gi_c2, d_gi_p1, d_gi_r1, d_gi_c1, d_dummy]:
            cuda_net.free_gpu(p)

    # --- Epoch Stats ---
    avg_train_loss = train_loss_sum / train_total
    avg_train_acc = train_correct / train_total
    
    # --- Validation ---
    val_loss, val_acc = evaluate(testloader)
    
    # --- Log ---
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)
    history['test_loss'].append(val_loss)
    history['test_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | LR: {new_lr:.5f} | "
          f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc*100:.2f}% | "
          f"Test Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

cuda_net.destroy_context()

# ================= 绘图 =================
plt.figure(figsize=(12, 5))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['test_loss'], label='Test Loss', linestyle='--')
plt.title('Loss Curve (Dropout + Cosine)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Acc Curve
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['test_acc'], label='Test Acc', linestyle='--')
plt.title('Accuracy Curve (Dropout + Cosine)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('result_dropout_cosine.png')
plt.show()

print(f"训练结束。最终测试集准确率: {history['test_acc'][-1]*100:.2f}%")