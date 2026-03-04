# Final Project for course AI_Programming

得分：19.5/20

## Task1

使⽤ PyTorch 中的卷积神经⽹络 (CNN) 来完成 CIFAR-10 数据集的图像分类。

### 运行代码

```
cd Task1
python HW1.py
```

## Task2

使用PyTorch的数据并行机制提高CIFAR-10训练的效率并进行对比。

### 运行代码

1. 确保安装了`torch`和`torchvision`
2. 切换并行/单卡：如果要运行双卡模式，请确保`torch.cuda.device_count() > 1`。若要强制单卡运行，可以手动注释`model = nn.DataParallel(model)`
3. 执行命令：`python task2.py`

## Task3

基于本学期前几次作业完成的cuda算子，基于cuda、pybind11、python等语言自主实现一个简单卷积网络，完成CIFAR-10数据集的图像分类任务。

在作业的基础上增加了Dropout和BN层，使用DeepVGG网络进行训练达到86%正确率，单epoch耗时15s左右

### 运行代码

```
cd Task3
python setup.py develop
python train5.py
```
