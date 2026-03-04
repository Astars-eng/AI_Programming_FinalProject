import numpy as np
import cuda_net

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

    def to_cpu(self):
        return cuda_net.to_cpu(self.ptr, list(self.shape))

    def zero_grad(self):
        if self.grad_ptr:
            size = int(np.prod(self.shape))
            pass 

class SGD:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=5e-4):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay # 新增参数
        # 初始化动量缓存
        self.velocities = [np.zeros(p.shape, dtype=np.float32) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad_ptr:
                # 1. 获取梯度 (GPU -> CPU)
                grad = cuda_net.to_cpu(p.grad_ptr, list(p.shape))
                
                # 2. 获取参数 (GPU -> CPU)
                p_data = cuda_net.to_cpu(p.ptr, list(p.shape))
                
                # 3. 应用 Weight Decay: grad = grad + decay * weight
                if self.weight_decay > 0:
                    grad += self.weight_decay * p_data
                
                # 4. 更新动量 v = m * v + grad
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                
                # 5. 更新参数 p = p - lr * v
                p_data = p_data - self.lr * self.velocities[i]
                
                # 6. 写回 GPU
                cuda_net.free_gpu(p.ptr)
                p.ptr = cuda_net.to_gpu(p_data)