# 🧠 LeNet 图像分类实验（CIFAR-10）
基于 PyTorch 复现 LeNet-5 卷积神经网络，在 CIFAR-10 数据集上完成图像分类任务。

## 📌 实验简介
LeNet 是早期经典卷积网络，用来学习图像局部特征；CIFAR-10 包含 60000 张 32×32 彩色图，一共 10 个类别（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）。
本实验：数据集加载 → 网络搭建 → 训练 → 验证 → 结果评估。

## 🛠️ 环境依赖
```bash
pip install torch torchvision matplotlib
