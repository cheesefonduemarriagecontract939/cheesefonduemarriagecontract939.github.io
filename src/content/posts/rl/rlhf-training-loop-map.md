---
title: "RLHF 训练闭环梳理：从 SFT 到 PPO / GRPO"
description: "先把后训练链路拆开，再决定每一步该读什么、记什么、复现什么。"
pubDatetime: 2026-07-10T09:00:00+08:00
featured: true
tags:
  - RLHF
  - Post-Training
  - PPO
  - GRPO
  - LLM
---

最近重新梳理后训练流程时，我发现最容易混乱的不是某个具体算法，而是**整条链路到底该怎么分层理解**。如果一开始就把 reward model、policy optimization、data filtering、sampling 全部堆在一起，读 paper 和看代码时会非常碎。

所以我现在会先把 RLHF 主线拆成四层：

## 1. 数据层

- SFT 数据从哪里来
- preference data 的分布是否稳定
- 正负样本是否和目标任务一致

如果这一层没有想清楚，后面 reward model 再强也只是把噪声拟合得更漂亮。

## 2. 评估层

这一层回答的问题是：**模型到底在什么意义上“更好”了？**

常见指标会混在一起：

- 任务成功率
- 奖励模型分数
- 长度偏置
- 拒答率
- 人工偏好对比

我更倾向先把“训练时优化的指标”和“真正关心的产品目标”严格分开，否则很容易出现 reward hacking。

## 3. 优化层

在这里才轮到 PPO、GRPO、DPO 这些方法。

我会先问三个问题：

1. 这类方法是否显式依赖 reward model？
2. 它对采样和 rollout 的成本有多敏感？
3. 它在长回答、长 reasoning 链条上会不会更不稳定？

把这三个问题放在前面，很多方法对比就不会停留在“谁更新”更 advanced，而会回到实际训练预算与稳定性。

## 4. 工程层

真正落地时，很多差异不是公式，而是工程细节：

- rollout 并发策略
- reference model 的缓存方式
- logprob 对齐
- advantage 归一化
- mixed precision 下的数值稳定性

这也是后面我想重点记录的部分，因为它们最适合做成可以回查的短笔记。

## 当前结论

如果只是想尽快建立整体图景，我觉得最有效的顺序不是“先啃算法”，而是：

1. 先画出训练链路图
2. 再明确每一层要解决的问题
3. 最后才进入 PPO / GRPO / DPO 的细节差异

后面我会把这篇作为总索引，继续往下挂 reward model、rollout 和 preference optimization 的分支笔记。
