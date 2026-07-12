---
author: Bakuma-sea
pubDatetime: 2026-07-12T23:50:00+08:00
title: GRPO 及其改进算法全景梳理
featured: true
tags:
  - GRPO
  - RL
  - Post-Training
  - LLM
description: 系统梳理 GRPO 之后的主要改进算法、核心问题、损失函数与工程改进路线。
timezone: Asia/Shanghai
---

# GRPO 及其改进算法全景梳理

> 本文系统梳理 GRPO 之后的主要改进算法，逐一分析每种算法相对于 GRPO 的核心改进点，并给出完整的损失函数公式。

---

## 0. 基线：GRPO（Group Relative Policy Optimization）

### 核心思想

GRPO 是 DeepSeek 在 DeepSeekMath 中提出的算法，核心创新是**去掉 Critic/Value 模型**，用同一问题的多个采样奖励的均值和标准差来计算优势函数。

### 损失函数

$$
\mathcal{L}_{\text{GRPO}}(\theta) = - \frac{1}{G} \sum_{i=1}^G \left[ \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( r_{i,t} \hat{A}_i, \ \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \cdot \mathbb{D}_{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right] \right]
$$

其中：

$$
r_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}, \quad \hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}
$$

**KL 散度为 Sequence Level**（整个序列的 KL，而非 token 级）：

$$
\mathbb{D}_{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right] = \frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - \log \frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - 1
$$

实际实现中，序列级 KL 通过 token 级 k3 估计器求和得到：

$$
\mathbb{D}_{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right] = \sum_{t=1}^{|o_i|} \left( \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1 \right)
$$

> **重要说明**：KL 散度项在损失函数中位于 **sequence level**（在 $\frac{1}{|o_i|}\sum_t$ 之外），而非 token level。这是 DeepSeekMath 原始论文和 DeepSeek-R1 采用的形式。KL 散度作为序列级的正则化项，约束新策略不偏离参考策略太远。

### GRPO 的核心问题

1. **Token 级重要性采样导致高方差**：奖励是序列级的，但重要性采样在 token 级进行，优化单位与奖励单位不匹配
2. **长度偏差**：长序列的梯度被 token 级平均稀释，导致短正确答案获得更大梯度，长错误答案惩罚不足
3. **难度偏差**：标准差极小的问题（全对/全错）在策略更新中被赋予过高权重
4. **熵崩溃**：对称裁剪限制了低概率 token 的探索空间，导致策略快速收敛到确定性策略
5. **零梯度问题**：当某问题的所有采样输出都正确或都错误时，优势为零，梯度消失
6. **MoE 模型不稳定**：token 级路由变化导致重要性比率剧烈波动

---

## 1. DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）

> 论文：DAPO: An Open-Source LLM Reinforcement Learning System at Scale
> 机构：字节跳动 & 清华大学 AIR
> 时间：2025.03

### 改进点

DAPO 在 GRPO 框架内做了四项工程改进，**未改变核心优化目标的形式**：

| 改进 | 解决的问题 | 具体做法 |
|------|-----------|---------|
| **Clip-Higher（解耦裁剪）** | 熵崩溃 | 将对称裁剪 $[1-\epsilon, 1+\epsilon]$ 解耦为 $[\epsilon_{\text{low}}, \epsilon_{\text{high}}]$，增大 $\epsilon_{\text{high}}$ 为低概率 token 留出更大增长空间 |
| **Dynamic Sampling（动态采样）** | 零梯度问题 | 过度采样 prompt，过滤掉奖励全为 1 或全为 0 的样本，只保留有效梯度的样本 |
| **Token-Level Loss（Token 级损失）** | 长度偏差 | 对所有 token 一起求平均（而非先在序列内平均再在序列间平均），让长序列的每个 token 都有公平的贡献 |
| **Overlong Reward Shaping（超长奖励调整）** | 过长响应截断引入的奖励噪声 | 对超过最大长度的响应施加 soft 惩罚，响应越长惩罚越大（详见下文） |

### Overlong Reward Shaping 详解

在 RL 训练中，通常设定最大生成长度 $L_{\text{max}}$。当响应超过此长度时会被截断，截断后的答案可能不完整，导致奖励信号不准确（奖励噪声）。DAPO 通过**超长奖励塑形**解决此问题：

**1. 长度奖励函数**（论文公式 13）：

$$
R_{\text{length}}(y) = \begin{cases} 0, & |y| \leq L_{\text{max}} - L_{\text{cache}} \\ \frac{(L_{\text{max}} - L_{\text{cache}}) - |y|}{L_{\text{cache}}}, & L_{\text{max}} - L_{\text{cache}} < |y| \leq L_{\text{max}} \\ -1, & |y| > L_{\text{max}} \end{cases}
$$

其中：
- $|y|$：响应长度（token 数）
- $L_{\text{max}}$：最大生成长度
- $L_{\text{cache}}$：软惩罚缓冲区长度

**2. 总奖励**：

$$
R_{\text{total}} = R_{\text{accuracy}} + R_{\text{length}}
$$

**3. 三段式惩罚机制**：

| 响应长度区间 | 奖励值 | 说明 |
|-------------|--------|------|
| $\vert y \vert \leq L_{\text{max}} - L_{\text{cache}}$ | $0$ | 安全区，无惩罚 |
| $L_{\text{max}} - L_{\text{cache}} < \vert y \vert \leq L_{\text{max}}$ | $[-1, 0)$ | 软惩罚区，线性递增惩罚 |
| $\vert y \vert > L_{\text{max}}$ | $-1$ | 截断区，最大惩罚 |

**4. 设计动机**：
- 直接对截断样本施加 $-1$ 惩罚会错误惩罚那些推理过程合理但只是过长的样本
- 软惩罚区让模型学到"尽量在限制内完成"，而非"长就是错"
- 避免模型对自身推理能力产生误解

**5. 超长样本过滤**（Overlong Filtering）：
- 对截断样本（达到 $L_{\text{max}}$ 的响应）屏蔽其损失
- 实验表明这显著稳定了训练并提升了性能

**6. 实际参数**（DAPO 论文）：
- $L_{\text{max}} = 16,384$ tokens
- $L_{\text{cache}} = 4,096$ tokens
- 最大生成 token 数 = $20,480$

### 损失函数

$$
\mathcal{L}_{\text{DAPO}}(\theta) = - \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \left[ \min \left( r_{i,t} \hat{A}_i, \ \text{clip}(r_{i,t}, \epsilon_{\text{low}}, \epsilon_{\text{high}}) \hat{A}_i \right) \right] + \frac{\beta}{G} \sum_{i=1}^G \mathbb{D}_{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right]
$$

其中 $\mathbb{D}_{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right]$ 为序列级 KL 散度（同 GRPO）。

**与 GRPO 的关键区别**：
- 分母从 $\frac{1}{G} \cdot \frac{1}{|o_i|}$ 变为 $\frac{1}{\sum_{i=1}^G |o_i|}$（全局 token 级平均）
- 裁剪范围从对称 $[1-\epsilon, 1+\epsilon]$ 变为非对称 $[\epsilon_{\text{low}}, \epsilon_{\text{high}}]$
- KL 散度同样为 sequence level

### 代码实现

```python
import torch

def dapo_loss(new_log_probs, old_log_probs, ref_log_probs, advantages, 
              response_mask, epsilon_low=0.8, epsilon_high=0.28, beta=0.01):
    """
    DAPO 损失函数
    
    Args:
        new_log_probs: 当前策略 log 概率 [batch, seq_len]
        old_log_probs: 旧策略 log 概率 [batch, seq_len]
        ref_log_probs: 参考策略 log 概率 [batch, seq_len]
        advantages: 优势值 [batch, 1]
        response_mask: 有效 token 掩码 [batch, seq_len]
        epsilon_low: 裁剪下限 (默认 0.8)
        epsilon_high: 裁剪上限 (默认 0.28，大于标准 0.2)
        beta: KL 系数
    """
    # 重要性采样比率
    log_ratios = new_log_probs - old_log_probs
    ratios = torch.exp(log_ratios)
    
    # 解耦裁剪（非对称）
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - epsilon_low, 1 + epsilon_high) * advantages
    
    # 策略损失
    policy_loss = -torch.min(surr1, surr2)
    
    # 策略损失：全局 token 级平均
    policy_loss = (policy_loss * response_mask).sum() / response_mask.sum()
    
    # KL 散度（sequence level：先 token 级计算 k3，再对序列求和）
    kl_per_token = (ref_log_probs.exp() / new_log_probs.exp()) - (ref_log_probs - new_log_probs) - 1
    # 对每条序列的 token 求和得到序列级 KL
    seq_lengths = response_mask.sum(dim=1, keepdim=True)  # [batch, 1]
    kl_per_seq = (kl_per_token * response_mask).sum(dim=1, keepdim=True)  # [batch, 1]
    kl_loss = kl_per_seq.mean()
    
    total_loss = policy_loss + beta * kl_loss
    
    return total_loss


def dynamic_sampling(prompts, model, reward_fn, G=64, max_attempts=3):
    """
    动态采样：过滤全对/全错的样本
    
    Args:
        prompts: prompt 列表
        model: 策略模型
        reward_fn: 奖励函数
        G: 每个 prompt 的目标采样数
        max_attempts: 最大尝试次数
    """
    valid_samples = []
    
    for prompt in prompts:
        for attempt in range(max_attempts):
            # 过度采样
            responses = model.generate(prompt, n=G*2)
            rewards = reward_fn(prompt, responses)
            
            # 过滤全对/全错
            if rewards.mean() > 0 and rewards.mean() < 1:
                # 保留有效样本
                valid_samples.append((prompt, responses, rewards))
                break
        else:
            # 如果多次尝试都全对/全错，跳过该 prompt
            continue
    
    return valid_samples


def overlong_reward_shaping(response_lengths, max_length=16384, cache_length=4096):
    """
    DAPO 超长奖励塑形
    
    Args:
        response_lengths: 每条响应的长度 [batch]
        max_length: 最大生成长度 L_max
        cache_length: 软惩罚缓冲区长度 L_cache
    
    Returns:
        length_rewards: 长度奖励 [batch]
    """
    safe_zone = max_length - cache_length  # 安全区上限
    
    rewards = torch.zeros_like(response_lengths, dtype=torch.float)
    
    # 安全区：无惩罚
    safe_mask = response_lengths <= safe_zone
    rewards[safe_mask] = 0.0
    
    # 软惩罚区：线性递增惩罚 (-1, 0)
    soft_mask = (response_lengths > safe_zone) & (response_lengths <= max_length)
    rewards[soft_mask] = (safe_zone - response_lengths[soft_mask]) / cache_length
    
    # 截断区：最大惩罚 -1
    over_mask = response_lengths > max_length
    rewards[over_mask] = -1.0
    
    return rewards


def compute_dapo_reward(accuracy_rewards, response_lengths, max_length=16384, cache_length=4096):
    """
    计算 DAPO 总奖励 = 准确率奖励 + 长度奖励
    
    Args:
        accuracy_rewards: 准确率奖励 [batch] (1 或 -1)
        response_lengths: 响应长度 [batch]
        max_length: 最大生成长度
        cache_length: 软惩罚缓冲区长度
    
    Returns:
        total_rewards: 总奖励 [batch]
    """
    length_rewards = overlong_reward_shaping(response_lengths, max_length, cache_length)
    total_rewards = accuracy_rewards + length_rewards
    return total_rewards
```

---

## 2. Dr.GRPO（Dynamic Reward GRPO）

> 论文：Understanding R1-Zero-Like Training: A Critical Perspective
> 机构：oat-zero 团队
> 时间：2025.03

### 改进点

Dr.GRPO 发现了 GRPO 的两个偏差问题，并提出了简洁的修正：

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| **响应长度偏差** | 序列内 token 平均导致短响应获得更大梯度 | 去掉 token 级平均，改为序列级损失 |
| **问题难度偏差** | 标准差极低的问题被赋予过高权重 | 去掉问题级标准差归一化 |

### 损失函数

$$
\mathcal{L}_{\text{Dr.GRPO}}(\theta) = - \frac{1}{G} \sum_{i=1}^G \left[ \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( r_{i,t} \hat{A}_i, \ \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right] - \frac{\beta}{G} \sum_{i=1}^G \mathbb{D}_{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right]
$$

其中优势计算也做了修正：

$$
\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{MAX\_TOKENS}} \quad \text{（用固定常数替代标准差）}
$$

**与 GRPO 的关键区别**：
- 优势计算的分母从 $\text{std}(\mathbf{r})$ 改为固定常数 $\text{MAX\_TOKENS}$（消除难度偏差）
- KL 散度同样为 sequence level

### 代码实现

```python
import torch

def drgrpo_loss(new_log_probs, old_log_probs, ref_log_probs, advantages, 
                response_mask, epsilon=0.2, beta=0.01, max_tokens=2048):
    """
    Dr.GRPO 损失函数
    
    关键区别：
    1. 不使用 token 级平均，使用固定常数 max_tokens 归一化
    2. 优势计算也使用 max_tokens 而非 std
    """
    # 重要性采样比率
    ratios = torch.exp(new_log_probs - old_log_probs)
    
    # 标准 clip
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2)
    
    # 策略损失：token 级平均
    policy_loss = (policy_loss * response_mask).sum(dim=1) / response_mask.sum(dim=1)
    policy_loss = policy_loss.mean()
    
    # KL 散度（sequence level）
    kl_per_token = (ref_log_probs.exp() / new_log_probs.exp()) - (ref_log_probs - new_log_probs) - 1
    kl_per_seq = (kl_per_token * response_mask).sum(dim=1)
    kl_loss = kl_per_seq.mean()
    
    total_loss = policy_loss + beta * kl_loss
    
    return total_loss


def drgrpo_advantage(rewards, max_tokens=2048):
    """
    Dr.GRPO 优势计算：使用固定常数替代标准差
    """
    rewards = torch.tensor(rewards, dtype=torch.float)
    # 关键区别：分母使用 max_tokens 而非 std
    advantages = (rewards - rewards.mean()) / max_tokens
    return advantages
```

---

## 3. GSPO（Group Sequence Policy Optimization）

> 论文：Group Sequence Policy Optimization
> 机构：Qwen 团队（阿里巴巴）
> 时间：2025.07
> 论文链接：https://arxiv.org/abs/2507.18071

### 改进点

GSPO 是 GRPO 改进中**真正改变优化粒度**的算法，将重要性采样从 token 级提升到序列级：

| 改进 | 说明 |
|------|------|
| **序列级重要性比率** | 用整个序列的似然比替代 token 级比率，消除 token 级噪声累积 |
| **长度归一化** | 对序列级比率取 $1/|o_i|$ 次幂，消除长度影响 |
| **序列级裁剪** | 裁剪作用于整个序列，而非单个 token |
| **天然适配 MoE** | 对单个 token 的似然不敏感，避免 MoE 路由变化导致的不稳定性 |

### 损失函数

$$
\mathcal{L}_{\text{GSPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \min \left( s_i(\theta) \hat{A}_i, \ \text{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right)
$$

其中**序列级重要性比率**：

$$
s_i(\theta) = \left( \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} \right)^{\frac{1}{|o_i|}} = \exp \left( \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \log \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} \right)
$$

**与 GRPO 的关键区别**：
- GRPO：每个 token 有独立的 $r_{i,t}$，每个 token 独立裁剪
- GSPO：整个序列共享一个 $s_i(\theta)$，所有 token 使用相同的裁剪决策

### 梯度分析

GSPO 的梯度（不考虑 clip）：

$$
\nabla_\theta J_{\text{GSPO}} = \frac{1}{G} \sum_{i=1}^G \hat{A}_i \cdot \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})
$$

GRPO 的梯度：

$$
\nabla_\theta J_{\text{GRPO}} = \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} r_{i,t} \hat{A}_i \cdot \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})
$$

**关键差异**：GSPO 对所有 token 赋予**相等权重**，而 GRPO 用 $r_{i,t}$ 加权，极端比率会导致梯度不稳定。

### 代码实现

```python
import torch

def gspo_loss(new_log_probs, old_log_probs, ref_log_probs, advantages, 
              response_mask, seq_lengths, epsilon=0.2, beta=0.01):
    """
    GSPO 损失函数：序列级重要性采样
    
    Args:
        new_log_probs: 当前策略 log 概率 [batch, seq_len]
        old_log_probs: 旧策略 log 概率 [batch, seq_len]
        ref_log_probs: 参考策略 log 概率 [batch, seq_len]
        advantages: 优势值 [batch]
        response_mask: 有效 token 掩码 [batch, seq_len]
        seq_lengths: 每条序列的实际长度 [batch]
        epsilon: 裁剪参数
        beta: KL 系数
    """
    batch_size = new_log_probs.shape[0]
    
    # 计算每个 token 的 log 比率
    token_log_ratios = new_log_probs - old_log_probs  # [batch, seq_len]
    
    # 序列级 log 比率：对有效 token 求平均
    # s_i = exp((1/|o_i|) * sum_t log(pi_new/pi_old))
    seq_log_ratios = (token_log_ratios * response_mask).sum(dim=1) / seq_lengths
    seq_ratios = torch.exp(seq_log_ratios)  # [batch]
    
    # 序列级裁剪
    surr1 = seq_ratios * advantages
    surr2 = torch.clamp(seq_ratios, 1 - epsilon, 1 + epsilon) * advantages
    
    # 策略损失（序列级）
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # KL 散度（token 级计算，然后序列级平均）
    kl = (ref_log_probs.exp() / new_log_probs.exp()) - (ref_log_probs - new_log_probs) - 1
    kl_loss = (kl * response_mask).sum(dim=1) / seq_lengths
    kl_loss = kl_loss.mean()
    
    total_loss = policy_loss + beta * kl_loss
    
    return total_loss
```

### GSPO-token 变体（适配多轮场景）

$$
\mathcal{L}_{\text{GSPO-token}}(\theta) = \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( s_{i,t}(\theta) \hat{A}_{i,t}, \ \text{clip}(s_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t} \right)
$$

其中：

$$
s_{i,t}(\theta) = \text{sg}[s_i(\theta)] \cdot \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\text{sg}[\pi_\theta(o_{i,t} \mid q, o_{i,<t})]}
$$

$\text{sg}$ 表示 stop gradient（只取数值，不计算梯度）。

---

## 4. GMPO（Geometric-Mean Policy Optimization）

> 论文：Geometric-Mean Policy Optimization
> 机构：UCAS（中国科学院大学）
> 时间：2025.07
> 论文链接：https://arxiv.org/abs/2507.20673

### 改进点

GMPO 与 GSPO 殊途同归，同样解决了 token 级重要性采样的不稳定性：

| 改进 | 说明 |
|------|------|
| **几何平均替代算术平均** | 用几何平均计算序列级重要性比率，对极端值更鲁棒 |
| **更大的裁剪范围** | 可使用更大的 $\epsilon$（如 0.4）进行更多策略探索 |
| **Token 级裁剪** | 保留 token 级裁剪的稳定性，同时用几何平均控制整体比率 |

### 损失函数

$$
\mathcal{L}_{\text{GMPO}}(\theta) = - \frac{1}{G} \sum_{i=1}^G \hat{A}_i \cdot \exp \left( \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \text{sgn}(\hat{A}_i) \cdot \Delta_{i,t}, \ \text{clip}(\text{sgn}(\hat{A}_i) \cdot \Delta_{i,t}, -\epsilon, \epsilon) \right) \right)
$$

其中 $\Delta_{i,t} = \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) - \log \pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})$。

**与 GSPO 的关键区别**：
- GSPO：先求平均，再裁剪
- GMPO：先对每个 token 裁剪，再求几何平均

### 代码实现

```python
import torch

def gmpo_loss(new_log_probs, old_log_probs, advantages, response_mask, epsilon=0.4):
    """
    GMPO 损失函数：几何平均 + token 级裁剪
    
    Args:
        new_log_probs: 当前策略 log 概率 [batch, seq_len]
        old_log_probs: 旧策略 log 概率 [batch, seq_len]
        advantages: 优势值 [batch, 1]
        response_mask: 有效 token 掩码 [batch, seq_len]
        epsilon: 裁剪参数（可取较大值，如 0.4）
    """
    # 计算 log 比率
    log_ratios = new_log_probs - old_log_probs  # [batch, seq_len]
    
    # 符号处理：advantage 为正时希望 ratio 增大，为负时希望 ratio 减小
    sgn_A = torch.sign(advantages)  # [batch, 1]
    
    # 对符号化后的 log 比率进行裁剪
    sgn_log_ratios = sgn_A * log_ratios
    clipped_sgn_log_ratios = torch.clamp(sgn_log_ratios, -epsilon, epsilon)
    
    # 取 min（保守更新）
    min_sgn_log_ratios = torch.min(sgn_log_ratios, clipped_sgn_log_ratios)
    
    # 恢复符号
    min_log_ratios = sgn_A * min_sgn_log_ratios
    
    # 几何平均：对有效 token 求平均后取 exp
    seq_log_ratios = (min_log_ratios * response_mask).sum(dim=1) / response_mask.sum(dim=1)
    importance_sampling_ratio = torch.exp(seq_log_ratios)
    
    # 损失
    loss = -(advantages.squeeze() * importance_sampling_ratio).mean()
    
    return loss
```

---

## 5. GFPO（Group Filtered Policy Optimization）

> 论文：Sample More to Think Less: Group Filtered Policy Optimization for Concise Reasoning
> 机构：微软研究院
> 时间：2025.08
> 论文链接：https://arxiv.org/abs/2508.09726

### 改进点

GFPO 解决 GRPO 的**响应长度膨胀**问题，可同时优化多个属性（如准确度和简洁性）：

| 改进 | 说明 |
|------|------|
| **数据过滤** | 在大规模采样池中筛选符合目标属性的响应，再进行策略优化 |
| **多属性优化** | 无需复杂奖励工程，通过过滤隐式实现多目标优化 |
| **兼容性强** | 干预在优势估计层面，可与任何 GRPO 变体（DAPO、Dr.GRPO 等）兼容 |

### 损失函数

$$
\mathcal{L}_{\text{GFPO}}(\theta) = - \frac{1}{G'} \sum_{i \in \mathcal{F}} \left[ \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( r_{i,t} \hat{A}_i^{\text{filtered}}, \ \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i^{\text{filtered}} \right) - \beta \cdot \mathbb{D}_{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right] \right]
$$

其中 $\mathcal{F}$ 是过滤后的响应集合，$G' = |\mathcal{F}|$。

**过滤条件**：
- 长度过滤：$|o_i| \leq L_{\max}$
- 效率过滤：$r_i / |o_i| \geq \eta_{\min}$

**过滤后的优势计算**：

$$
\hat{A}_i^{\text{filtered}} = \frac{r_i - \text{mean}(\mathbf{r}_{\mathcal{F}})}{\text{std}(\mathbf{r}_{\mathcal{F}})}
$$

### 代码实现

```python
import torch

def gfpo_filter(responses, rewards, max_length=None, min_efficiency=None):
    """
    GFPO 数据过滤
    
    Args:
        responses: 响应列表
        rewards: 奖励列表
        max_length: 最大长度限制
        min_efficiency: 最小效率阈值（奖励/token数）
    
    Returns:
        filtered_indices: 过滤后的索引
    """
    G = len(responses)
    mask = torch.ones(G, dtype=torch.bool)
    
    # 长度过滤
    if max_length is not None:
        lengths = torch.tensor([len(r) for r in responses])
        mask &= (lengths <= max_length)
    
    # 效率过滤
    if min_efficiency is not None:
        lengths = torch.tensor([len(r) for r in responses], dtype=torch.float)
        efficiency = torch.tensor(rewards) / lengths
        mask &= (efficiency >= min_efficiency)
    
    return torch.where(mask)[0]


def gfpo_advantage(rewards, filtered_indices):
    """
    GFPO 优势计算：仅在过滤后的样本上计算
    """
    filtered_rewards = torch.tensor([rewards[i] for i in filtered_indices])
    mean = filtered_rewards.mean()
    std = filtered_rewards.std()
    advantages = (filtered_rewards - mean) / (std + 1e-8)
    return advantages


def gfpo_loss(new_log_probs, old_log_probs, ref_log_probs, rewards, 
              response_mask, max_length=2048, min_efficiency=0.01,
              epsilon=0.2, beta=0.01):
    """
    GFPO 损失函数
    """
    # 数据过滤
    seq_lengths = response_mask.sum(dim=1)
    filtered_indices = gfpo_filter_with_tensors(
        seq_lengths, rewards, max_length, min_efficiency
    )
    
    if len(filtered_indices) == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    # 过滤后的数据
    filtered_new_log_probs = new_log_probs[filtered_indices]
    filtered_old_log_probs = old_log_probs[filtered_indices]
    filtered_ref_log_probs = ref_log_probs[filtered_indices]
    filtered_mask = response_mask[filtered_indices]
    
    # 过滤后的优势
    filtered_advantages = gfpo_advantage_with_tensors(
        rewards[filtered_indices]
    )
    
    # 标准 GRPO 损失（在过滤后的数据上）
    ratios = torch.exp(filtered_new_log_probs - filtered_old_log_probs)
    surr1 = ratios * filtered_advantages.unsqueeze(1)
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * filtered_advantages.unsqueeze(1)
    policy_loss = -torch.min(surr1, surr2)
    
    # 策略损失：token 级平均
    policy_loss = (policy_loss * filtered_mask).sum() / filtered_mask.sum()
    
    # KL 散度（sequence level）
    kl_per_token = (filtered_ref_log_probs.exp() / filtered_new_log_probs.exp()) - \
                  (filtered_ref_log_probs - filtered_new_log_probs) - 1
    kl_per_seq = (kl_per_token * filtered_mask).sum(dim=1)
    kl_loss = kl_per_seq.mean()
    
    total_loss = policy_loss + beta * kl_loss
    
    return total_loss
```

---

## 6. SAPO（Soft Adaptive Policy Optimization）

> 论文：Soft Adaptive Policy Optimization
> 机构：Qwen 团队（阿里巴巴）
> 时间：2025.11
> 论文链接：https://arxiv.org/abs/2511.20347

### 改进点

SAPO 用**平滑的软门控机制**替代硬裁剪，兼具序列一致性与 token 级自适应性：

| 改进 | 说明 |
|------|------|
| **软门控替代硬裁剪** | 用 Sigmoid 函数构建连续信任区域，避免梯度突变 |
| **自适应温度** | 温度参数 $\tau$ 自动调节门控的平滑程度 |
| **非对称温度** | 正负 token 使用不同温度，加速抑制高方差负样本 |
| **序列一致性 + Token 自适应** | 保持 GSPO 的序列级稳定性，同时保留 token 级灵活性 |

### 损失函数

$$
\mathcal{L}_{\text{SAPO}}(\theta) = - \frac{1}{G} \sum_{i=1}^G \left[ \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} g_{i,t}(\theta) \cdot r_{i,t} \cdot \hat{A}_i - \beta \cdot \mathbb{D}_{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right] \right]
$$

其中**软门控函数**：

$$
g_{i,t}(\theta) = \sigma \left( \frac{\log r_{i,t}}{\tau_{i,t}} \right) = \frac{1}{1 + \exp(-\log r_{i,t} / \tau_{i,t})}
$$

**自适应温度参数**：

$$
\tau_{i,t} = \begin{cases} \tau_{\text{pos}} & \text{if } \hat{A}_i > 0 \\ \tau_{\text{neg}} & \text{if } \hat{A}_i < 0 \end{cases}
$$

通常 $\tau_{\text{neg}} < \tau_{\text{pos}}$，对负样本使用更低的温度（更激进的裁剪）。

**与硬裁剪的对比**：

| 特性 | 硬裁剪 (GRPO/GSPO) | 软门控 (SAPO) |
|------|-------------------|--------------|
| 梯度连续性 | 在裁剪点不连续 | 处处连续可导 |
| 异常 token 处理 | 整个序列被抑制 | 仅降低异常 token 权重 |
| 样本效率 | 低（有用信号被丢弃） | 高（保留近策略 token 信号） |
| 超参数敏感性 | 高（裁剪范围固定） | 低（温度自适应调节） |

### 代码实现

```python
import torch
import torch.nn.functional as F

def sapo_loss(new_log_probs, old_log_probs, ref_log_probs, advantages, 
              response_mask, tau_pos=0.5, tau_neg=0.2, beta=0.01):
    """
    SAPO 损失函数：软自适应门控
    
    Args:
        new_log_probs: 当前策略 log 概率 [batch, seq_len]
        old_log_probs: 旧策略 log 概率 [batch, seq_len]
        ref_log_probs: 参考策略 log 概率 [batch, seq_len]
        advantages: 优势值 [batch, 1]
        response_mask: 有效 token 掩码 [batch, seq_len]
        tau_pos: 正样本温度（较大，更平滑）
        tau_neg: 负样本温度（较小，更激进）
        beta: KL 系数
    """
    # 重要性采样比率
    log_ratios = new_log_probs - old_log_probs
    ratios = torch.exp(log_ratios)
    
    # 自适应温度：正负样本使用不同温度
    advantages_expanded = advantages.expand_as(log_ratios)
    tau = torch.where(advantages_expanded > 0, 
                      torch.tensor(tau_pos), 
                      torch.tensor(tau_neg))
    
    # 软门控函数：sigmoid(log_ratio / tau)
    # 当 ratio ≈ 1 时，gate ≈ 0.5；当 ratio 远离 1 时，gate → 0 或 1
    gate = torch.sigmoid(log_ratios / tau)
    
    # 策略损失：用软门控替代硬裁剪
    policy_loss = -gate * ratios * advantages_expanded
    
    # 策略损失：token 级平均
    policy_loss = (policy_loss * response_mask).sum(dim=1) / response_mask.sum(dim=1)
    policy_loss = policy_loss.mean()
    
    # KL 散度（sequence level）
    kl_per_token = (ref_log_probs.exp() / new_log_probs.exp()) - (ref_log_probs - new_log_probs) - 1
    kl_per_seq = (kl_per_token * response_mask).sum(dim=1)
    kl_loss = kl_per_seq.mean()
    
    total_loss = policy_loss + beta * kl_loss
    
    return total_loss
```

---

## 7. DUPO（Dynamic Universal Policy Optimization）

> 相关论文：多种动态采样策略的统称
> 时间：2025

### 改进点

DUPO 的核心思想是**动态重复采样**，提高样本利用率：

| 改进 | 说明 |
|------|------|
| **重复采样** | 对同一 prompt 多次采样，选择最有信息量的响应 |
| **动态批次构建** | 根据奖励分布动态调整采样策略 |
| **加速训练** | 减少无效样本的计算浪费 |

### 损失函数

DUPO 的损失函数形式与 GRPO 相同，区别在于**采样策略**：

$$
\mathcal{L}_{\text{DUPO}}(\theta) = - \frac{1}{G} \sum_{i=1}^G \left[ \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( r_{i,t} \hat{A}_i, \ \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \cdot \mathbb{D}_{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right] \right]
$$

**采样策略**：

$$
\mathcal{D}_{\text{DUPO}} = \text{TopK}(\{ (q, o_i, r_i) \}_{i=1}^{N}, k=G, \text{criterion}=\text{variance})
$$

即从 $N$ 个采样中选择方差最大的 $G$ 个，确保每个 batch 都有充分的梯度信号。

### 代码实现

```python
import torch

def dupo_sampling(prompts, model, reward_fn, N=128, G=64):
    """
    DUPO 动态重复采样
    
    Args:
        prompts: prompt 列表
        model: 策略模型
        reward_fn: 奖励函数
        N: 每个 prompt 的总采样数
        G: 每个 prompt 选择的样本数
    
    Returns:
        selected_samples: 选中的样本
    """
    all_samples = []
    
    for prompt in prompts:
        # 大量采样
        responses = model.generate(prompt, n=N)
        rewards = reward_fn(prompt, responses)
        
        # 选择方差最大的 G 个样本
        rewards_tensor = torch.tensor(rewards)
        
        # 计算每个子集的方差（简化：选择奖励分布最分散的 G 个）
        # 实际实现可能更复杂，如选择包含正负样本的子集
        sorted_indices = torch.argsort(rewards_tensor)
        
        # 选择两端的样本（确保有正有负）
        half_G = G // 2
        selected_indices = torch.cat([
            sorted_indices[:half_G],  # 最低奖励
            sorted_indices[-half_G:]  # 最高奖励
        ])
        
        selected_samples.append({
            'prompt': prompt,
            'responses': [responses[i] for i in selected_indices],
            'rewards': [rewards[i] for i in selected_indices]
        })
    
    return selected_samples
```

---

## 8. 其他值得关注的改进算法

### 8.1 SRPO（Sample Reweighted Policy Optimization）

利用历史重采样机制，保留关键问题（关键样本），以便在后续训练阶段继续使用。

### 8.2 OPO（Optimal baseline Policy Optimization）

采用最优奖励基线设计，最小化梯度方差，提高训练稳定性。

### 8.3 EMPO（Entropy-guided Model-based Policy Optimization）

将语义熵引入优化目标，评估不确定性并将其纳入优势值计算。

### 8.4 AAPO（Advantage Accumulation Policy Optimization）

提出融合优势动量的优势估计方法，使策略更新更平滑、更有效。

### 8.5 BNPO（Beta Normalized Policy Optimization）

使用具有动态更新参数的 Beta 分布对奖励进行自适应归一化。

### 8.6 COPO（Consistency-Aware Policy Optimization）

基于结果一致性引入结构化全局奖励，解决多个回答收敛到相同结果导致优势为 0 的问题。

---

## 9. 算法演进总结

### 演进脉络

```
GRPO (DeepSeek, 2024.02)
  │
  ├── DAPO (ByteDance, 2025.03) ── 工程改进：解耦裁剪、动态采样、Token级损失
  │
  ├── Dr.GRPO (oat-zero, 2025.03) ── 理论分析：消除长度偏差和难度偏差
  │
  ├── GSPO (Qwen, 2025.07) ── 范式转变：Token级 → 序列级重要性采样
  │     │
  │     └── SAPO (Qwen, 2025.11) ── 软门控替代硬裁剪
  │
  ├── GMPO (UCAS, 2025.07) ── 几何平均 + 更大裁剪范围
  │
  ├── GFPO (Microsoft, 2025.08) ── 数据过滤实现多属性优化
  │
  └── DUPO (多种, 2025) ── 动态重复采样提高样本效率
```

### 核心改进维度对比

| 算法 | 优化粒度 | 裁剪机制 | 采样策略 | KL 处理 | 核心贡献 |
|------|---------|---------|---------|---------|---------|
| **GRPO** | Token 级 | 对称硬裁剪 | 固定 G | **序列级** k3 估计器 | 去掉 Critic |
| **DAPO** | Token 级 | 非对称硬裁剪 | 动态过滤 | **序列级** k3 估计器 | 工程优化集大成 |
| **Dr.GRPO** | Token 级 | 对称硬裁剪 | 固定 G | **序列级** k3 估计器 | 消除长度/难度偏差 |
| **GSPO** | 序列级 | 序列级硬裁剪 | 固定 G | 序列级 KL | 优化粒度转变 |
| **GMPO** | Token 级 | Token 级硬裁剪 | 固定 G | 无 | 几何平均 + 宽裁剪 |
| **GFPO** | Token 级 | 对称硬裁剪 | 过滤采样 | **序列级** k3 估计器 | 多属性优化 |
| **SAPO** | Token 级 | 软门控 | 固定 G | **序列级** k3 估计器 | 平滑信任区域 |
| **DUPO** | Token 级 | 对称硬裁剪 | 动态重复 | **序列级** k3 估计器 | 样本效率提升 |

### 损失函数统一形式

所有 GRPO 变体都可以写成以下统一形式：

$$
\mathcal{L}(\theta) = - \frac{1}{\mathcal{N}} \sum_{i} \left[ \sum_{t} w_{i,t} \cdot f(r_{i,t}, \hat{A}_i) \right] - \frac{\beta}{G} \sum_{i} \mathbb{D}_{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right]
$$

> **注意**：所有算法的 KL 散度均为 **sequence level**（在 token 级求和之外），这是 DeepSeekMath 和 DeepSeek-R1 的标准做法。

其中：

| 算法 | $w_{i,t}$ | $f(r, A)$ | $\mathcal{N}$ |
|------|-----------|-----------|---------------|
| GRPO | $\frac{1}{G} \cdot \frac{1}{\|o_i\|}$ | $\min(rA, \text{clip}(r, 1-\epsilon, 1+\epsilon)A)$ | $G \cdot \|o_i\|$ |
| DAPO | $\frac{1}{\sum \|o_i\|}$ | $\min(rA, \text{clip}(r, \epsilon_{\text{low}}, \epsilon_{\text{high}})A)$ | $\sum \|o_i\|$ |
| Dr.GRPO | $\frac{1}{G}$ | $\min(rA, \text{clip}(r, 1-\epsilon, 1+\epsilon)A)$ | $G$ |
| GSPO | $\frac{1}{G}$ | $\min(s_i A, \text{clip}(s_i, 1-\epsilon, 1+\epsilon)A)$ | $G$ |
| GMPO | $\frac{1}{G}$ | $\exp(\text{mean}(\min(\text{sgn}(A)\Delta, \text{clip}(\text{sgn}(A)\Delta, -\epsilon, \epsilon)))) \cdot A$ | $G$ |
| GFPO | $\frac{1}{G'} \cdot \frac{1}{\|o_i\|}$ | $\min(rA, \text{clip}(r, 1-\epsilon, 1+\epsilon)A)$ | $G' \cdot \|o_i\|$ |
| SAPO | $\frac{1}{G} \cdot \frac{1}{\|o_i\|}$ | $\sigma(\log r / \tau) \cdot r \cdot A$ | $G \cdot \|o_i\|$ |

---

## 10. 实践建议

### 场景选择指南

| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| 快速复现/基线 | GRPO | 简单、广泛支持 |
| 长 CoT 推理 | DAPO 或 GSPO | 解决长度偏差和熵崩溃 |
| MoE 模型 | GSPO | 天然适配 MoE 路由变化 |
| 需要控制响应长度 | GFPO | 多属性过滤 |
| 追求最高稳定性 | SAPO | 软门控最平滑 |
| 样本效率优先 | DUPO | 动态采样减少浪费 |

### 关键超参数建议

| 参数 | GRPO | DAPO | GSPO | SAPO |
|------|------|------|------|------|
| $\epsilon$ | 0.2 | $\epsilon_{\text{low}}=0.8, \epsilon_{\text{high}}=0.28$ | 0.2 | - |
| $\tau$ | - | - | - | $\tau_{\text{pos}}=0.5, \tau_{\text{neg}}=0.2$ |
| $\beta$ | 0.01 | 0.01 | 0.01 | 0.01 |
| G | 64 | 64（动态过滤） | 64 | 64 |

---

## 参考资料

- DeepSeekMath: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DAPO: https://arxiv.org/abs/2503.14476
- Dr.GRPO: https://arxiv.org/abs/2503.20783
- GSPO: https://arxiv.org/abs/2507.18071
- GMPO: https://arxiv.org/abs/2507.20673
- GFPO: https://arxiv.org/abs/2508.09726
- SAPO: https://arxiv.org/abs/2511.20347
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
- verl: https://github.com/volcengine/verl
