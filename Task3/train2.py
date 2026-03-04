import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from autograd import Tensor 
import cuda_net 

# ================= 配置 =================
BATCH_SIZE = 64
LR = 0.01 
MOMENTUM = 0.9
EPOCHS = 40 
DATA_ROOT = './data'

# ================= 数据增强 =================
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

trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ================= 辅助：Kaiming 初始化 =================
def kaiming_init(shape):
    fan_in = np.prod(shape[1:])
    bound = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape).astype(np.float32) * bound

# ================= 定义网络 =================
cuda_net.init_context()

print("初始化网络参数...")
# Layer 1
w1 = Tensor(kaiming_init((32, 3, 3, 3)), requires_grad=True)
b1 = Tensor(np.zeros(32, dtype=np.float32), requires_grad=True)

# Layer 2
w2 = Tensor(kaiming_init((64, 32, 3, 3)), requires_grad=True)
b2 = Tensor(np.zeros(64, dtype=np.float32), requires_grad=True)

# Layer 3 (FC)
w3 = Tensor(kaiming_init((256, 64 * 8 * 8)), requires_grad=True)
b3 = Tensor(np.zeros(256, dtype=np.float32), requires_grad=True)

# Layer 4 (Output)
w4 = Tensor(kaiming_init((10, 256)), requires_grad=True)
b4 = Tensor(np.zeros(10, dtype=np.float32), requires_grad=True)

# 定义带 Momentum 的 SGD
class MomentumSGD:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros(p.shape, dtype=np.float32) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad_ptr:
                grad = cuda_net.to_cpu(p.grad_ptr, list(p.shape))
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                p_data = cuda_net.to_cpu(p.ptr, list(p.shape))
                p_data -= self.lr * self.velocities[i]
                cuda_net.free_gpu(p.ptr)
                p.ptr = cuda_net.to_gpu(p_data)

optimizer = MomentumSGD([w1, b1, w2, b2, w3, b3, w4, b4], lr=LR, momentum=MOMENTUM)

# ================= 训练函数 =================
def forward_pass(inputs_ptr, N, training=True):
    # --- Block 1 ---
    c1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
    cuda_net.forward_conv(inputs_ptr, c1, w1.ptr, b1.ptr, N, 3, 32, 32, 32)
    
    r1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
    cuda_net.forward_relu(c1, r1, N * 32 * 32 * 32)
    
    p1 = cuda_net.alloc_gpu(N * 32 * 16 * 16)
    m1 = cuda_net.alloc_gpu(N * 32 * 32 * 32) # Mask
    cuda_net.forward_pool(r1, p1, m1, N, 32, 32, 32)
    
    # --- Block 2 ---
    c2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
    cuda_net.forward_conv(p1, c2, w2.ptr, b2.ptr, N, 32, 16, 16, 64)
    
    r2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
    cuda_net.forward_relu(c2, r2, N * 64 * 16 * 16)
    
    p2 = cuda_net.alloc_gpu(N * 64 * 8 * 8)
    m2 = cuda_net.alloc_gpu(N * 64 * 16 * 16) # Mask
    cuda_net.forward_pool(r2, p2, m2, N, 64, 16, 16)
    
    # --- FC 1 ---
    fc1 = cuda_net.alloc_gpu(N * 256)
    cuda_net.forward_fc(p2, fc1, w3.ptr, b3.ptr, N, 64*8*8, 256)
    
    r3 = cuda_net.alloc_gpu(N * 256) # ReLU after FC1
    cuda_net.forward_relu(fc1, r3, N * 256)

    # --- FC 2 (Output) ---
    out = cuda_net.alloc_gpu(N * 10)
    cuda_net.forward_fc(r3, out, w4.ptr, b4.ptr, N, 256, 10)

    # Cache for backward
    cache = (c1, r1, p1, m1, c2, r2, p2, m2, fc1, r3, out)
    return out, cache

def free_cache(cache):
    for ptr in cache:
        cuda_net.free_gpu(ptr)

# ================= 评估函数 =================
def evaluate(loader):
    correct = 0
    total = 0
    for inputs, labels in loader:
        N = inputs.shape[0]
        inputs_np = inputs.numpy()
        labels_np = labels.numpy().astype(np.int32)
        
        d_in = cuda_net.to_gpu(inputs_np)
        
        logits_ptr, cache = forward_pass(d_in, N, training=False)
        
        logits = cuda_net.to_cpu(logits_ptr, [N, 10])
        preds = np.argmax(logits, axis=1)
        correct += np.sum(preds == labels_np)
        total += N
        
        cuda_net.free_gpu(d_in)
        free_cache(cache)
        
    return correct / total

# ================= 训练 Loop =================
print("开始训练...")
history = {
    'train_loss': [], 
    'train_acc': [], 
    'val_acc': []
}

for epoch in range(EPOCHS):
    # 记录时间
    start_time = time.time()
    
    total_loss = 0
    train_correct = 0
    train_total = 0
    
    for i, (inputs, labels) in enumerate(trainloader):
        N = inputs.shape[0]
        inputs_np = inputs.numpy()
        labels_np = labels.numpy().astype(np.int32)
        d_in = cuda_net.to_gpu(inputs_np)
        
        # 1. Forward
        logits_ptr, cache = forward_pass(d_in, N, training=True)
        (c1, r1, p1, m1, c2, r2, p2, m2, fc1, r3, out) = cache
        
        # 2. Loss
        d_grad_logits = cuda_net.alloc_gpu(N * 10)
        loss = cuda_net.cross_entropy(logits_ptr, labels_np, d_grad_logits, N, 10)
        
        # --- [新增] 计算训练集准确率 ---
        # 需要把 logits 拷回 CPU
        cpu_logits = cuda_net.to_cpu(logits_ptr, [N, 10])
        preds = np.argmax(cpu_logits, axis=1)
        train_correct += np.sum(preds == labels_np)
        train_total += N
        # -----------------------------
        
        # 3. Backward
        d_gi_r3 = cuda_net.alloc_gpu(N * 256)
        cuda_net.backward_fc(r3, logits_ptr, w4.ptr, b4.ptr, N, 256, 10, d_grad_logits, d_gi_r3, w4.grad_ptr, b4.grad_ptr)
        
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
        
        # 5. Clean up
        cuda_net.free_gpu(d_in)
        cuda_net.free_gpu(d_grad_logits)
        free_cache(cache)
        for p in [d_gi_r3, d_gi_fc1, d_gi_p2, d_gi_r2, d_gi_c2, d_gi_p1, d_gi_r1, d_gi_c1, d_dummy]:
            cuda_net.free_gpu(p)
            
        total_loss += loss

    # --- Epoch 结束统计 ---
    epoch_duration = time.time() - start_time
    avg_loss = total_loss / len(trainloader)
    train_acc = train_correct / train_total
    val_acc = evaluate(testloader)
    
    # 记录历史
    history['train_loss'].append(avg_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_duration:.1f}s | "
          f"Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {val_acc*100:.2f}%")

cuda_net.destroy_context()

# ================= 绘图与保存 (更新版) =================
plt.figure(figsize=(12, 5))

# 1. Loss Curve
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', color='blue')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 2. Accuracy Curve (Train vs Test)
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc', color='green', linestyle='--')
plt.plot(history['val_acc'], label='Test Acc', color='red')
plt.title('Accuracy: Train vs Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
save_path = 'training_curves.png'
plt.savefig(save_path)
print(f"结果曲线图已保存为 {save_path}")