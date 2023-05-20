

# Bridgeformer: Finetuning and Explaination for Bert and ViT in VQA Task

## Introduction

### 背景

* 多模态任务需要大模型，而预训练 + 微调的范式在多模态任务中也应用得较广

* 多模态模型也需要解释性
* VQA任务

### 挑战

* 没有工作提出针对微调多模态模型的可解释性

### 现有方法及问题

* 利用预训练模型进行简单的Finetuning
* 可解释性的欠缺

### 我们的方法与实验

<img src="idea.assets/image-20221123135341767.png" alt="image-20221123135341767" style="zoom:50%;" />

* 提出bridgeformer架构更好的融合模态交互
* 

### 我们的贡献

* 将NLP中的大模型微调方法应用于多模态预训练模型
* 提出了Bridgeformer架构成功结合了Bert和ViT
* 我们以可视化的方式事后解释了微调前后的模态交互变化，体现了参数化的微调方法带来了更好的模态融合

## Related Works

### VQA任务

### 多模态模型的可解释性

### 大模型微调方法

## Methodology

### Bridgeformer 架构

<img src="idea.assets/image-20221123140128041.png" alt="image-20221123140128041" style="zoom:50%;" />

### 在训练后的微调策略

### 可解释性的方法DIME的改进版

## Experiments

### 训练Bridgeformer

### 微调Bridgeformer

### Bridgeformer 内部可视化

## Analysis

### 微调前后的参数

### 内部可视化的分析

## Conclusion







