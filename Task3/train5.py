import time
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json # 用于保存训练数据

# 引入你的编译模块
import cuda_net 

# ================= 配置参数 =================
BATCH_SIZE = 64
EPOCHS = 50           # 建议至少跑50轮
LR_MAX = 0.02
LR_MIN = 0.0001
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DROPOUT_RATE = 0.5
BN_MOMENTUM = 0.1
BN_EPS = 1e-5

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
                if self.weight_decay > 0:
                    grad += self.weight_decay * p_data
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                p_data -= self.lr * self.velocities[i]
                cuda_net.free_gpu(p.ptr)
                p.ptr = cuda_net.to_gpu(p_data)

# ================= 数据准备 =================
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

# ================= 模型初始化 (3层卷积深度版) =================
cuda_net.init_context()

def kaiming_init(shape):
    fan_in = np.prod(shape[1:])
    return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)

print("初始化参数 (Deep VGG 3-Block)...")

# --- Block 1: 3 -> 32 ---
w1 = Tensor(kaiming_init((32, 3, 3, 3)), requires_grad=True)
b1 = Tensor(np.zeros(32), requires_grad=True)
gamma1 = Tensor(np.ones(32, dtype=np.float32), requires_grad=True)
beta1  = Tensor(np.zeros(32, dtype=np.float32), requires_grad=True)
rm1 = cuda_net.alloc_gpu(32); rv1 = cuda_net.to_gpu(np.ones(32, dtype=np.float32))

# --- Block 2: 32 -> 64 ---
w2 = Tensor(kaiming_init((64, 32, 3, 3)), requires_grad=True)
b2 = Tensor(np.zeros(64), requires_grad=True)
gamma2 = Tensor(np.ones(64, dtype=np.float32), requires_grad=True)
beta2  = Tensor(np.zeros(64, dtype=np.float32), requires_grad=True)
rm2 = cuda_net.alloc_gpu(64); rv2 = cuda_net.to_gpu(np.ones(64, dtype=np.float32))

# --- Block 3: 64 -> 128 (关键升级) ---
w3_conv = Tensor(kaiming_init((128, 64, 3, 3)), requires_grad=True)
b3_conv = Tensor(np.zeros(128), requires_grad=True)
gamma3 = Tensor(np.ones(128, dtype=np.float32), requires_grad=True)
beta3  = Tensor(np.zeros(128, dtype=np.float32), requires_grad=True)
rm3 = cuda_net.alloc_gpu(128); rv3 = cuda_net.to_gpu(np.ones(128, dtype=np.float32))

# --- FC Layers ---
w_fc1 = Tensor(kaiming_init((256, 128 * 4 * 4)), requires_grad=True)
b_fc1 = Tensor(np.zeros(256), requires_grad=True)
w_fc2 = Tensor(kaiming_init((10, 256)), requires_grad=True)
b_fc2 = Tensor(np.zeros(10), requires_grad=True)

params = [w1, b1, gamma1, beta1, w2, b2, gamma2, beta2, w3_conv, b3_conv, gamma3, beta3, w_fc1, b_fc1, w_fc2, b_fc2]
optimizer = SGD(params, lr=LR_MAX, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# ================= 前向传播 (含 BN & Dropout) =================
def forward_pass(inputs_ptr, N, training=True):
    # Block 1
    c1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
    cuda_net.forward_conv(inputs_ptr, c1, w1.ptr, b1.ptr, N, 3, 32, 32, 32)
    bn1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
    sm1, sv1 = cuda_net.alloc_gpu(32), cuda_net.alloc_gpu(32)
    if training: cuda_net.forward_bn_train(c1, bn1, sm1, sv1, rm1, rv1, gamma1.ptr, beta1.ptr, N, 32, 32, 32, BN_MOMENTUM, BN_EPS)
    else: cuda_net.forward_bn_test(c1, bn1, rm1, rv1, gamma1.ptr, beta1.ptr, N, 32, 32, 32, BN_EPS)
    r1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
    cuda_net.forward_relu(bn1, r1, N * 32 * 32 * 32)
    p1 = cuda_net.alloc_gpu(N * 32 * 16 * 16)
    m1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
    cuda_net.forward_pool(r1, p1, m1, N, 32, 32, 32)
    
    # Block 2
    c2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
    cuda_net.forward_conv(p1, c2, w2.ptr, b2.ptr, N, 32, 16, 16, 64)
    bn2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
    sm2, sv2 = cuda_net.alloc_gpu(64), cuda_net.alloc_gpu(64)
    if training: cuda_net.forward_bn_train(c2, bn2, sm2, sv2, rm2, rv2, gamma2.ptr, beta2.ptr, N, 64, 16, 16, BN_MOMENTUM, BN_EPS)
    else: cuda_net.forward_bn_test(c2, bn2, rm2, rv2, gamma2.ptr, beta2.ptr, N, 64, 16, 16, BN_EPS)
    r2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
    cuda_net.forward_relu(bn2, r2, N * 64 * 16 * 16)
    p2 = cuda_net.alloc_gpu(N * 64 * 8 * 8)
    m2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
    cuda_net.forward_pool(r2, p2, m2, N, 64, 16, 16)

    # Block 3
    c3 = cuda_net.alloc_gpu(N * 128 * 8 * 8)
    cuda_net.forward_conv(p2, c3, w3_conv.ptr, b3_conv.ptr, N, 64, 8, 8, 128)
    bn3 = cuda_net.alloc_gpu(N * 128 * 8 * 8)
    sm3, sv3 = cuda_net.alloc_gpu(128), cuda_net.alloc_gpu(128)
    if training: cuda_net.forward_bn_train(c3, bn3, sm3, sv3, rm3, rv3, gamma3.ptr, beta3.ptr, N, 128, 8, 8, BN_MOMENTUM, BN_EPS)
    else: cuda_net.forward_bn_test(c3, bn3, rm3, rv3, gamma3.ptr, beta3.ptr, N, 128, 8, 8, BN_EPS)
    r3 = cuda_net.alloc_gpu(N * 128 * 8 * 8)
    cuda_net.forward_relu(bn3, r3, N * 128 * 8 * 8)
    p3 = cuda_net.alloc_gpu(N * 128 * 4 * 4)
    m3 = cuda_net.alloc_gpu(N * 128 * 8 * 8)
    cuda_net.forward_pool(r3, p3, m3, N, 128, 8, 8)
    
    # FC Block
    fc1 = cuda_net.alloc_gpu(N * 256)
    cuda_net.forward_fc(p3, fc1, w_fc1.ptr, b_fc1.ptr, N, 128*4*4, 256)
    r_fc1 = cuda_net.alloc_gpu(N * 256)
    cuda_net.forward_relu(fc1, r_fc1, N * 256)
    d_out = cuda_net.alloc_gpu(N * 256)
    d_mask = cuda_net.alloc_gpu(N * 256)
    if training: cuda_net.forward_dropout(r_fc1, d_out, d_mask, N * 256, DROPOUT_RATE)
    else: cuda_net.forward_dropout(r_fc1, d_out, d_mask, N * 256, 0.0)
    out = cuda_net.alloc_gpu(N * 10)
    cuda_net.forward_fc(d_out, out, w_fc2.ptr, b_fc2.ptr, N, 256, 10)
    
    cache = (c1, bn1, sm1, sv1, r1, p1, m1, c2, bn2, sm2, sv2, r2, p2, m2, c3, bn3, sm3, sv3, r3, p3, m3, fc1, r_fc1, d_out, d_mask)
    return out, cache

# ================= 评估函数 =================
def evaluate(loader):
    correct = 0; total = 0
    for inputs, labels in loader:
        N = inputs.shape[0]
        d_in = cuda_net.to_gpu(inputs.numpy())
        labels_np = labels.numpy().astype(np.int32)
        logits_ptr, cache = forward_pass(d_in, N, training=False)
        logits = cuda_net.to_cpu(logits_ptr, [N, 10])
        correct += np.sum(np.argmax(logits, axis=1) == labels_np)
        total += N
        cuda_net.free_gpu(d_in); cuda_net.free_gpu(logits_ptr)
        for ptr in cache: cuda_net.free_gpu(ptr)
    return correct / total

# ================= 训练循环 =================
history = {'loss': [], 'train_acc': [], 'test_acc': []} # 新增 train_acc
print("开始训练...")
start_time = time.time()

for epoch in range(EPOCHS):
    new_lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * epoch / EPOCHS))
    optimizer.lr = new_lr
    
    train_loss = 0.0
    train_correct = 0  # 新增
    train_total = 0    # 新增
    
    for i, (inputs, labels) in enumerate(trainloader):
        N = inputs.shape[0]; inputs_np = inputs.numpy(); labels_np = labels.numpy().astype(np.int32)
        
        # 1. Forward
        d_in = cuda_net.to_gpu(inputs_np)
        logits_ptr, cache = forward_pass(d_in, N, training=True)
        
        # Unpack
        (c1, bn1, sm1, sv1, r1, p1, m1, c2, bn2, sm2, sv2, r2, p2, m2, c3, bn3, sm3, sv3, r3, p3, m3, fc1, r_fc1, d_out, d_mask) = cache
        
        # 2. Loss & Acc Calculation
        d_grad_logits = cuda_net.alloc_gpu(N * 10)
        loss = cuda_net.cross_entropy(logits_ptr, labels_np, d_grad_logits, N, 10)
        
        # --- 计算训练集准确率 (新增) ---
        # 拷贝 Logits 到 CPU 进行 Argmax
        logits_cpu = cuda_net.to_cpu(logits_ptr, [N, 10])
        preds = np.argmax(logits_cpu, axis=1)
        train_correct += np.sum(preds == labels_np)
        train_total += N
        # ----------------------------
        
        # 3. Backward
        d_gi_drop = cuda_net.alloc_gpu(N * 256)
        cuda_net.backward_fc(d_out, logits_ptr, w_fc2.ptr, b_fc2.ptr, N, 256, 10, d_grad_logits, d_gi_drop, w_fc2.grad_ptr, b_fc2.grad_ptr)
        d_gi_r_fc1 = cuda_net.alloc_gpu(N * 256)
        cuda_net.backward_dropout(d_gi_drop, d_gi_r_fc1, d_mask, N * 256, DROPOUT_RATE)
        d_gi_fc1 = cuda_net.alloc_gpu(N * 256)
        cuda_net.backward_relu(fc1, d_gi_r_fc1, d_gi_fc1, N * 256)
        d_gi_p3 = cuda_net.alloc_gpu(N * 128*4*4)
        cuda_net.backward_fc(p3, fc1, w_fc1.ptr, b_fc1.ptr, N, 128*4*4, 256, d_gi_fc1, d_gi_p3, w_fc1.grad_ptr, b_fc1.grad_ptr)
        
        d_gi_r3 = cuda_net.alloc_gpu(N * 128 * 8 * 8)
        cuda_net.backward_pool(d_gi_p3, m3, d_gi_r3, N, 128, 8, 8)
        d_gi_bn3 = cuda_net.alloc_gpu(N * 128 * 8 * 8)
        cuda_net.backward_relu(bn3, d_gi_r3, d_gi_bn3, N * 128 * 8 * 8)
        d_gi_c3 = cuda_net.alloc_gpu(N * 128 * 8 * 8)
        cuda_net.backward_bn(d_gi_bn3, c3, d_gi_c3, gamma3.grad_ptr, beta3.grad_ptr, sm3, sv3, gamma3.ptr, N, 128, 8, 8, BN_EPS)
        
        d_gi_p2 = cuda_net.alloc_gpu(N * 64 * 8 * 8)
        cuda_net.backward_conv(p2, w3_conv.ptr, d_gi_c3, d_gi_p2, w3_conv.grad_ptr, b3_conv.grad_ptr, N, 64, 8, 8, 128)
        d_gi_r2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
        cuda_net.backward_pool(d_gi_p2, m2, d_gi_r2, N, 64, 16, 16)
        d_gi_bn2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
        cuda_net.backward_relu(bn2, d_gi_r2, d_gi_bn2, N * 64 * 16 * 16)
        d_gi_c2 = cuda_net.alloc_gpu(N * 64 * 16 * 16)
        cuda_net.backward_bn(d_gi_bn2, c2, d_gi_c2, gamma2.grad_ptr, beta2.grad_ptr, sm2, sv2, gamma2.ptr, N, 64, 16, 16, BN_EPS)
        
        d_gi_p1 = cuda_net.alloc_gpu(N * 32 * 16 * 16)
        cuda_net.backward_conv(p1, w2.ptr, d_gi_c2, d_gi_p1, w2.grad_ptr, b2.grad_ptr, N, 32, 16, 16, 64)
        d_gi_r1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
        cuda_net.backward_pool(d_gi_p1, m1, d_gi_r1, N, 32, 32, 32)
        d_gi_bn1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
        cuda_net.backward_relu(bn1, d_gi_r1, d_gi_bn1, N * 32 * 32 * 32)
        d_gi_c1 = cuda_net.alloc_gpu(N * 32 * 32 * 32)
        cuda_net.backward_bn(d_gi_bn1, c1, d_gi_c1, gamma1.grad_ptr, beta1.grad_ptr, sm1, sv1, gamma1.ptr, N, 32, 32, 32, BN_EPS)
        d_dummy = cuda_net.alloc_gpu(N * 3 * 32 * 32)
        cuda_net.backward_conv(d_in, w1.ptr, d_gi_c1, d_dummy, w1.grad_ptr, b1.grad_ptr, N, 3, 32, 32, 32)

        optimizer.step()
        
        # Free
        ptrs = [d_in, d_grad_logits, logits_ptr, d_gi_drop, d_gi_r_fc1, d_gi_fc1, d_gi_p3, d_gi_r3, d_gi_bn3, d_gi_c3, d_gi_p2, d_gi_r2, d_gi_bn2, d_gi_c2, d_gi_p1, d_gi_r1, d_gi_bn1, d_gi_c1, d_dummy]
        for p in ptrs: cuda_net.free_gpu(p)
        for p in cache: cuda_net.free_gpu(p)
        train_loss += loss

    # --- Epoch Summary ---
    avg_loss = train_loss / len(trainloader)
    avg_train_acc = train_correct / train_total # 训练集准确率
    val_acc = evaluate(testloader)
    
    history['loss'].append(avg_loss)
    history['train_acc'].append(avg_train_acc)
    history['test_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | LR: {new_lr:.5f} | "
          f"Loss: {avg_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}% | Test Acc: {val_acc*100:.2f}%")

cuda_net.destroy_context()

# ================= 结果绘图与保存 (更新版) =================
# 1. 保存原始数据
with open('training_history.json', 'w') as f:
    json.dump(history, f)

# 2. 绘制双子图 (Subplots)
plt.figure(figsize=(12, 5))

# 左图: Loss
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss', color='tab:blue')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 右图: Accuracy (Train vs Test)
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc', color='tab:green', linestyle='--')
plt.plot(history['test_acc'], label='Test Acc', color='tab:orange', linewidth=2)
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0) # 固定Y轴范围0-1
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
# 直接保存图片
plt.savefig('deep_vgg_metrics.png', dpi=120)
print("\n训练完成！结果已保存为 'deep_vgg_metrics.png' 和 'training_history.json'")