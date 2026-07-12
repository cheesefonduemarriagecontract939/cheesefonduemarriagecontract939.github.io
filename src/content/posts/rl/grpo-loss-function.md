---
author: Bakuma-sea
pubDatetime: 2026-07-12T23:00:00+08:00
title: 手撕 GRPO 损失函数
featured: false
tags:
  - GRPO
  - RL
  - Loss Function
  - LLM
description: 从公式组成、策略梯度项、KL 约束与优势函数出发，逐步拆解 GRPO 损失函数。
timezone: Asia/Shanghai
---

# 手撕GRPO损失函数

> 本文整理自知乎文章《【手撕LLM-GRPO】你只管给Reward，剩下的交给RL》
> 作者：小冬瓜AIGC
> 原文链接：https://zhuanlan.zhihu.com/p/20812786520

---

## 第一部分：GRPO损失函数讲解

### 1. GRPO算法概述

GRPO（Group Relative Policy Optimization，群组相对策略优化）是 DeepSeek 提出的一种用于大语言模型强化学习训练的算法，首次出现在 DeepSeekMath 论文中。其核心特点：

- **仅需两个模型**：目标优化模型（policy model）和参考模型（reference model）
- **不需要 Reward Model**：通过规则奖励（rule-based reward）直接判别
- **组内相对排名替代价值估计**：不需要 Critic/Value Model，解决了 PPO 的内存瓶颈

GRPO 的优化过程：

1. 给定问题 q，让模型采样生成 G 个回答（通常 G=64）
2. 通过规则奖励函数对每个回答打分
3. 根据组内奖励的均值和标准差计算优势值
4. 结合重要性采样和 KL 散度约束更新策略

### 2. 核心损失函数

GRPO 的完整损失函数如下：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = - \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min \left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,< t})} \hat{A}_{i,t}, \, \text{clip}\left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,< t})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_{i,t} \right) - \beta \mathbb{D}_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right] \right]
$$

其中：

$$
\mathbb{D}_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right] = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,< t})}{\pi_\theta(o_{i,t} \mid q, o_{i,< t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,< t})}{\pi_\theta(o_{i,t} \mid q, o_{i,< t})} - 1
$$

$$
\hat{A}_{i,t} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}
$$

### 3. 损失函数组成分析

GRPO 损失函数由三部分组成：

| 组成部分 | 说明 |
|---------|------|
| **策略梯度项（Policy Gradient）** | 通过重要性采样比率和优势函数驱动策略更新，使用 clip 限制更新幅度 |
| **KL 散度项** | 约束新策略不偏离参考策略太远，采用无偏小方差的 KL 估计器 |
| **长度归一化** | 对每条回答的 token 损失求平均，消除长度影响 |

#### 3.1 策略梯度项

$$
\min \left( r_t \cdot \hat{A}_t, \, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot \hat{A}_t \right)
$$

其中 $r_t = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,< t})}$ 是新旧策略的比率。

- 当优势 $\hat{A}_t > 0$ 时（好回答），增加该回答的概率
- 当优势 $\hat{A}_t < 0$ 时（差回答），降低该回答的概率
- clip 操作限制比率在 $[1-\epsilon, 1+\epsilon]$ 范围内，防止过大的策略更新

#### 3.2 优势函数

$$
\hat{A}_{i,t} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}
$$

- $\mathbf{r} = \{r_1, r_2, \ldots, r_G\}$ 是一组采样回答的奖励
- 优势是 **sentence-level** 的，即一个回答中每个 token 的优势值相同
- 优势通过组内均值和标准差归一化，高于平均的获得正激励，低于平均的获得负激励
- 当全对或全错时，优势为 0，梯度无效，可以 skip 掉该样本

#### 3.3 KL 散度项

GRPO 采用了一种优化的 KL 估计器（k3 估计器）：

$$
\mathbb{D}_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right] = (\gamma - 1) - \log \gamma, \quad \gamma = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,t})}
$$

与标准 KL 散度 $\mathbb{D}_{\text{KL}} = -\log\frac{\pi_{\text{ref}}}{\pi_\theta}$ 相比，k3 估计器具有**无偏且方差小**的特性：

| 估计器 | 公式 | 偏差 | 方差 |
|--------|------|------|------|
| k1（naive） | $-\log r$ | 无偏 | 很大 |
| k2（low variance） | $\frac{1}{2}(\log r)^2$ | 有偏 | 较小 |
| **k3（unbiased low variance）** | $(r-1) - \log r$ | **无偏** | **较小** |

k3 估计器保证所有值为正，且 $r-1 > \log r$，有效降低了方差。

### 4. 奖励函数

GRPO 通常使用规则奖励（rule-based reward），不需要训练额外的 Reward Model：

- **准确性奖励（Accuracy Reward）**：判断模型输出答案与标签是否一致，一致为 1，否则为 0
- **格式奖励（Format Reward）**：判断模型是否遵循指定输出格式（如 `<think>...</think><answer>...</answer>`），遵循为 1，否则为 0

奖励是 sentence-level 的标量值，即 $\mathbf{r} = \{r_1, r_2, \ldots, r_G\} \in \{0, 1\}$。

### 5. 关于 KL 项的讨论

在 TRL 项目中，有实验表明可以去除 KL 项（将 $\beta$ 置为 0），这样做的好处：

- 不需要加载 ref-model，减少显存占用
- 减少一次前向 ref_policy 的计算
- 没有 KL 约束，参数优化更自由，更容易探索到好的回答
- 可以通过梯度裁剪（max_grad_norm）来避免不稳定性

但去除 KL 项也可能导致策略偏离过远，需要根据具体任务权衡。

---

## 第二部分：手撕代码

### 1. 优势函数实现

```python
import torch

def grpo_advantage(rewards):
    """
    计算GRPO的组内相对优势
    
    Args:
        rewards: 一组采样回答的奖励列表，如 [1, 0, 0, 0, 1, 0]
    
    Returns:
        advantages: 每个回答的优势值
    """
    epsilon = 0.00001
    rewards = torch.tensor(rewards, dtype=torch.float)
    A = (rewards - rewards.mean()) / (rewards.std() + epsilon)
    return A


# 测试
if __name__ == "__main__":
    # Case 1: 全错
    A = grpo_advantage([0, 0, 0, 0, 0, 0])
    print("全错:", A)
    # 输出: tensor([-0., -0., -0., -0., -0., -0.])

    # Case 2: 1个回答正确
    A = grpo_advantage([1, 0, 0, 0, 0, 0])
    print("1个正确:", A)
    # 输出: tensor([2.0468, -0.4094, -0.4094, -0.4094, -0.4094, -0.4094])

    # Case 3: 2个回答正确
    A = grpo_advantage([1, 0, 0, 0, 1, 0])
    print("2个正确:", A)
    # 输出: tensor([1.2905, -0.6452, -0.6452, -0.6452, 1.2905, -0.6452])

    # Case 4: 全对
    A = grpo_advantage([1, 1, 1, 1, 1, 1])
    print("全对:", A)
    # 输出: tensor([0., 0., 0., 0., 0., 0.])

    # Case 5: 64个样本只有1个回答正确
    reward_batch = [0] * 64
    reward_batch[0] = 1
    A = grpo_advantage(reward_batch)
    print("64个样本1个正确:", A)
    # 正确项的优势约为 7.89，其余约为 -0.125
```

### 2. GRPO KL 散度实现

```python
import torch

def grpo_kl(pi_logprob, pi_ref_logprob):
    """
    计算GRPO的KL散度（k3估计器，无偏小方差）
    
    Args:
        pi_logprob: 当前策略的 log 概率
        pi_ref_logprob: 参考策略的 log 概率
    
    Returns:
        kl: KL散度值（均为正数）
    """
    return pi_ref_logprob.exp() / pi_logprob.exp() - (pi_ref_logprob - pi_logprob) - 1


# 测试
if __name__ == "__main__":
    pi = torch.randn(3, 5)       # batch=3, sequence=5
    pi_ref = torch.randn(3, 5)
    
    pi_logprob = torch.nn.functional.log_softmax(pi, dim=1)
    pi_ref_logprob = torch.nn.functional.log_softmax(pi_ref, dim=1)
    
    kl = grpo_kl(pi_logprob, pi_ref_logprob)
    print("KL散度:", kl)
    # 输出均为正数
```

### 3. 完整 GRPO Loss 实现

```python
import torch
import torch.nn.functional as F


def grpo_kl(pi_logprob, pi_ref_logprob):
    """GRPO KL散度（k3估计器）"""
    return pi_ref_logprob.exp() / pi_logprob.exp() - (pi_ref_logprob - pi_logprob) - 1


def grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob, advantage, input_len, len_oi,
              epsilon=0.2, beta=0.01, group_num=3):
    """
    计算GRPO损失函数
    
    Args:
        pi_logprob: 当前策略的 log 概率 [batch_size, seq_len]
        pi_old_logprob: 旧策略的 log 概率 [batch_size, seq_len]
        pi_ref_logprob: 参考策略的 log 概率 [batch_size, seq_len]
        advantage: 每个回答的优势值 [batch_size]
        input_len: 输入 prompt 的长度（用于 mask）
        len_oi: 每条回答的采样长度
        epsilon: clip 范围，默认 0.2
        beta: KL 项系数，默认 0.01
        group_num: 采样数量 G
    
    Returns:
        loss: GRPO 损失值
    """
    bs, seq_len = pi_logprob.shape
    
    # 设定 mask，仅对 response 部分计算 loss
    mask = torch.zeros(bs, seq_len)
    mask[:, input_len:] = 1
    
    # 计算新旧策略比率
    ratio = torch.exp(pi_logprob - pi_old_logprob)
    ratio_clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    
    # 扩展优势维度 [batch_size] -> [batch_size, 1]
    advantage = advantage.unsqueeze(dim=1)
    
    # 策略梯度项：取 min(未clip, clip) * advantage
    policy_gradient = torch.minimum(ratio * advantage, ratio_clip * advantage)
    
    # KL 散度项
    kl = grpo_kl(pi_logprob, pi_ref_logprob)
    
    # 组合损失
    loss = (policy_gradient - beta * kl) * mask
    
    # 归一化：除以 G 和每条回答的长度
    len_oi = torch.tensor([len_oi] * group_num, dtype=torch.long)
    loss = (-1 / group_num) * (1 / len_oi.unsqueeze(dim=1)) * loss
    loss = loss.sum()
    
    return loss


# 测试
if __name__ == "__main__":
    # 模拟输出分布
    pi_logits = torch.randn(3, 5, 32)      # batch=3, seq_len=5, vocab_size=32
    pi_ref_logits = torch.randn(3, 5, 32)
    pi_old_logits = torch.randn(3, 5, 32)
    
    # 获取 log prob
    pi_logprob = F.log_softmax(pi_logits, dim=-1)
    pi_ref_logprob = F.log_softmax(pi_ref_logits, dim=-1)
    pi_old_logprob = F.log_softmax(pi_old_logits, dim=-1)
    
    # 模拟 token ids（输入为 11,12,13，输出为后续 token）
    token_ids = torch.tensor([
        [11, 12, 13, 14, 15],
        [11, 12, 13, 15, 16],
        [11, 12, 13, 16, 17],
    ])
    
    # 根据实际 token ids 提取对应概率
    pi_logprob = torch.gather(pi_logprob, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    pi_ref_logprob = torch.gather(pi_ref_logprob, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    pi_old_logprob = torch.gather(pi_old_logprob, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    
    # 计算优势
    A = grpo_advantage([1, 0, 0])
    
    # 计算 loss
    loss = grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob, A, input_len=3, len_oi=2)
    print("GRPO Loss:", loss)
    # 输出: tensor(0.0033)
```

### 4. TRL 版本的 GRPO 实现分析

TRL（Transformer Reinforcement Learning）库中的 GRPO 实现做了简化：

```python
# TRL 简化版 GRPO Loss（trl <= 0.15.0）
advantages = inputs["advantages"]

# 关键代码：x - x.detach() 使得 ratio 恒为 1
per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
per_token_loss = -(per_token_loss - self.beta * per_token_kl)
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
```

**关键分析：**

1. `per_token_logps - per_token_logps.detach()` 前向时恒为 0，因此 `ratio = exp(0) = 1`
2. 反向传播时，`per_token_logps.detach()` 被视为常数，梯度仍然能通过 `per_token_logps` 正常回传
3. 这等价于 new-policy 只更新一次就完成当前批次的参数优化
4. 虽然简化了，但实验表明仍然有效

**与标准 GRPO 的区别：**

- 标准 GRPO：t=1 时用 $\theta_{t=1}$ 获取 old/new policy（ratio=1），更新后得到 $\theta_{t=2}$；t=2 时用 $\theta_{t=2}$ 获取 new policy，此时 ratio ≠ 1
- TRL 简化版：t=1 更新完梯度后即完成当前批次优化，不进行多轮迭代

### 5. KL 估计器对比验证

```python
import torch
import torch.distributions as dis

# 验证三种 KL 估计器的偏差和方差
p = dis.Normal(loc=0, scale=1)
q = dis.Normal(loc=0.1, scale=1)
x = q.sample(sample_shape=(10_000_000,))
truekl = dis.kl_divergence(p, q)

logr = p.log_prob(x) - q.log_prob(x)
k1 = -logr                    # naive estimator
k2 = logr ** 2 / 2            # low variance estimator
k3 = (logr.exp() - 1) - logr  # unbiased low variance estimator (GRPO使用)

print("True KL:", truekl.item())
for name, k in [("k1", k1), ("k2", k2), ("k3", k3)]:
    bias = (k.mean() - truekl) / truekl
    std = k.std() / truekl
    print(f"{name}: bias={bias.item():.4f}, std={std.item():.4f}")
```

**结果（q=N(0,1), p=N(0.1,1)，true KL=0.005）：**

| 估计器 | 偏差/真实值 | 标准差/真实值 |
|--------|------------|--------------|
| k1 | 0 | 20 |
| k2 | 0.002 | 1.42 |
| **k3** | **0** | **1.42** |

k3 估计器同时具备无偏性和低方差，是 GRPO 采用的形式。

---

## 总结

1. **GRPO 损失函数**由策略梯度项 + KL 散度项组成，通过组内相对优势驱动策略优化
2. **优势函数**通过组内奖励的均值和标准差归一化，sentence-level 粒度
3. **KL 项**采用 k3 估计器，无偏且方差小，约束策略不偏离参考分布太远
4. **代码实现**的核心是：计算 ratio → clip → 乘以优势 → 减去 KL → 按长度归一化
5. **TRL 简化版**通过 `x - x.detach()` 技巧使 ratio 恒为 1，虽然简化但仍能正常工作

---

## 参考资料

- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- John Schulman: Approximating KL Divergence
- Open-R1: https://github.com/huggingface/open-r1
- GRPO Loss 开源实现: https://github.com/dhcode-cpp/grpo-loss
