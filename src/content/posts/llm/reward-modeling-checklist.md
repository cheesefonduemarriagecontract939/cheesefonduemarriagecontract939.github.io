---
title: "Reward Model 阅读检查单"
description: "读 reward model 相关论文时，我想固定追问的不是架构，而是数据与误差。"
pubDatetime: 2026-07-08T10:30:00+08:00
featured: true
tags:
  - Reward Model
  - Preference Data
  - LLM
  - RLHF
---

reward model 的文章很容易越看越散，因为每篇 paper 都会强调不同的损失函数、采样策略或者标注协议。为了避免每次重头整理，我先给自己固定一份检查单。

## 先看数据，而不是先看模型

第一件事永远是问：

- preference data 从哪里来
- pairwise 比较是否足够一致
- 不同 annotator 之间偏好差异多大
- 数据是否和真实目标任务同分布

如果数据本身不稳定，后面的建模花样通常只是把问题包得更精致。

## 再看误差会如何传导

reward model 最大的问题之一是：它的误差不会停留在评估环节，而是会直接进入 policy optimization。

我尤其想追问：

- 偏差会不会被 rollout 放大
- 错误高分样本是否更容易被重复采到
- length bias 是被缓解了还是被强化了

## 最后看工程接口

reward model 真正接进训练系统时，关键不只是 accuracy，而是接口是否干净：

- logit 或 score 输出是否稳定
- batching 和 serving 代价有多高
- 是否需要额外 calibrate
- 更新节奏能否和 policy 训练解耦

## 暂时的阅读策略

后面再读相关 paper，我会优先在笔记里补齐三件事：

1. 数据假设
2. 误差传播路径
3. 工程接入成本

如果这三项说不清楚，我会默认自己还没有真正理解那篇文章。
