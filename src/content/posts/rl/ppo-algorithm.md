---
author: Bakuma-sea
pubDatetime: 2026-07-12T22:50:00+08:00
title: 手撕 PPO 算法：损失分析与代码实现
featured: false
tags:
  - PPO
  - RL
  - Policy Optimization
description: 从策略梯度、优势函数、重要性采样、TRPO 到 PPO Clip 目标，系统拆解 PPO 算法。
timezone: Asia/Shanghai
---

# 手撕PPO算法 —— 损失分析与代码实现

> 本文整理自学城、知乎、CSDN、SegmentFault 等多个技术社区的 PPO 相关文章，旨在帮助读者从零理解并实现 PPO 算法。

---

## 第一部分：损失分析

### 1. PPO 算法概述

PPO（Proximal Policy Optimization，近端策略优化）是 OpenAI 在 2017 年提出的一种策略梯度算法，目前已成为强化学习领域最主流的算法之一，也是大模型 RLHF 对齐的核心算法。

**核心思想**：通过限制策略更新的幅度，避免传统策略梯度方法中因参数更新过大导致的策略崩溃问题。

**PPO 的两个核心组件**：

| 组件 | 解决的问题 | 方法 |
|------|-----------|------|
| GAE（广义优势估计） | TD Error 估计不准（偏差-方差权衡） | 多步 TD Error 的指数衰减求和 |
| Clipped Surrogate Objective | 行为策略与目标策略分布差异过大 | 裁剪重要性采样比率 |

### 2. 从策略梯度到 PPO 的推导脉络

#### 2.1 基础策略梯度

策略梯度方法的目标是最大化期望回报：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} r_t \right]
$$

其梯度为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]
$$

其中 $G_t$ 是从时刻 $t$ 开始的累积回报。

**问题**：$G_t$ 的方差很大，导致训练不稳定。

#### 2.2 引入优势函数（Advantage）

用优势函数 $A(s, a) = Q(s, a) - V(s)$ 替代 $G_t$：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A(s_t, a_t) \right]
$$

优势函数衡量的是"在状态 $s_t$ 下选择动作 $a_t$ 比平均好多少"，方差更小。

#### 2.3 引入重要性采样（Off-Policy）

为了复用旧策略采集的数据，引入重要性采样比率：

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
$$

目标函数变为：

$$
J^{\text{IS}}(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \sum_{t=0}^{T} \rho_t(\theta) \cdot A(s_t, a_t) \right]
$$

**问题**：当新旧策略差异过大时，$\rho_t$ 可能变得非常大，导致梯度方差爆炸。

#### 2.4 TRPO：信赖域约束

TRPO 通过 KL 散度约束策略更新幅度：

$$
\max_\theta \quad \mathbb{E} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A(s, a) \right]
$$

$$
\text{s.t.} \quad \mathbb{E} \left[ D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s)) \right] \leq \delta
$$

**问题**：需要计算二阶导数（Hessian 矩阵），实现复杂，计算成本高。

#### 2.5 PPO：简化 TRPO

PPO 通过裁剪机制隐式地限制策略更新，避免了复杂的二阶优化。

### 3. GAE：广义优势估计

#### 3.1 TD Error（时序差分残差）

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

其中：
- $r_t$：时刻 $t$ 的即时奖励
- $\gamma$：折扣因子
- $V(s_t)$：Critic 网络估计的状态价值
- $\delta_t$：表示"实际回报比预期好多少"

#### 3.2 多步优势估计

| 步数 | 优势估计 | 偏差 | 方差 |
|------|---------|------|------|
| 1步 | $A_t^{(1)} = \delta_t$ | 大 | 小 |
| 2步 | $A_t^{(2)} = \delta_t + \gamma \delta_{t+1}$ | 中 | 中 |
| ∞步 | $A_t^{(\infty)} = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}$ | 小 | 大 |

#### 3.3 GAE 公式

GAE 是对多步优势的指数衰减加权：

$$
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中 $\lambda \in [0, 1]$ 是 GAE 参数：

- $\lambda = 0$：退化为 1 步 TD Error（偏差大，方差小）
- $\lambda = 1$：等价于蒙特卡洛估计（偏差小，方差大）
- 通常取 $\lambda = 0.95$，在偏差和方差之间取得平衡

#### 3.4 GAE 的递推形式（代码友好）

$$
\delta_t = r_t + \gamma V(s_{t+1}) \cdot (1 - d_t^{\text{done}}) - V(s_t)
$$

$$
A_t = \delta_t + \gamma \lambda A_{t+1} \cdot (1 - d_t^{\text{done}})
$$

**注意区分两个标志位**：
- $d_t^{\text{done}}$（dw）：回合真正结束（成功/失败），此时 $V(s_{t+1})$ 无意义，置零
- $d_t^{\text{truncated}}$（done）：广义结束（包括人为截断如超时），此时 $A_{t+1}$ 不再反向传播

### 4. PPO 的损失函数

PPO 的总损失由三部分组成：

$$
L^{\text{total}} = L^{\text{policy}} + c_1 L^{\text{value}} - c_2 L^{\text{entropy}}
$$

#### 4.1 策略损失（Clipped Surrogate Objective）

$$
L^{\text{CLIP}}(\theta) = -\mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \quad \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

其中重要性采样比率：

$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} = \exp \left( \log \pi_\theta(a_t | s_t) - \log \pi_{\theta_{\text{old}}}(a_t | s_t) \right)
$$

**裁剪机制的四种情况分析**：

| 情况 | 优势 $A_t$ | 比率 $r_t$ | 行为 |
|------|-----------|-----------|------|
| 1 | $A_t > 0$（好动作） | $r_t > 1+\epsilon$ | 裁剪，防止过度增大该动作概率 |
| 2 | $A_t > 0$（好动作） | $1 \leq r_t \leq 1+\epsilon$ | 不裁剪，正常增大 |
| 3 | $A_t < 0$（差动作） | $r_t < 1-\epsilon$ | 裁剪，防止过度减小该动作概率 |
| 4 | $A_t < 0$（差动作） | $1-\epsilon \leq r_t \leq 1$ | 不裁剪，正常减小 |

**核心直觉**：
- 当 $A_t > 0$ 时，我们希望增大 $\pi_\theta(a_t|s_t)$，但如果已经增大了很多（$r_t > 1+\epsilon$），就裁剪掉额外梯度，防止策略跑太远
- 当 $A_t < 0$ 时，我们希望减小 $\pi_\theta(a_t|s_t)$，但如果已经减小了很多（$r_t < 1-\epsilon$），就裁剪掉额外梯度
- 取 $\min$ 保证总是选择更保守的更新

#### 4.2 价值函数损失

$$
L^{\text{value}}(\phi) = \mathbb{E}_t \left[ \left( V_\phi(s_t) - V_t^{\text{target}} \right)^2 \right]
$$

其中目标值：

$$
V_t^{\text{target}} = A_t + V(s_t) = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} + V(s_t)
$$

等价于：

$$
V_t^{\text{target}} = \sum_{l=0}^{\infty} \gamma^l r_{t+l}
$$

即蒙特卡洛回报。

**注意**：实践中通常使用 Huber 损失（Smooth L1）替代 MSE，对异常值更鲁棒：

$$
L^{\text{value}}(\phi) = \mathbb{E}_t \left[ \text{SmoothL1}(V_\phi(s_t), V_t^{\text{target}}) \right]
$$

#### 4.3 熵正则化项

$$
L^{\text{entropy}}(\theta) = \mathbb{E}_t \left[ H(\pi_\theta(\cdot | s_t)) \right] = \mathbb{E}_t \left[ -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t) \right]
$$

**作用**：
- 鼓励探索，防止策略过早收敛到确定性策略
- 在损失函数中减去熵（即 $+ c_2 \cdot H$），相当于最大化熵，保持策略的随机性
- 训练初期可取较大系数（如 0.05），训练后期减小（如 0.001）

#### 4.4 完整损失函数

$$
L^{\text{total}} = \underbrace{- \mathbb{E} \left[ \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t) \right]}_{\text{策略损失}} + \underbrace{c_1 \cdot \mathbb{E} \left[ (V_\phi(s_t) - V_t^{\text{target}})^2 \right]}_{\text{价值损失}} - \underbrace{c_2 \cdot \mathbb{E} \left[ H(\pi_\theta) \right]}_{\text{熵正则}}
$$

典型超参数：$\epsilon = 0.2$, $c_1 = 0.5$, $c_2 = 0.01$

### 5. PPO 与相关算法的对比

| 算法 | 策略约束 | 偏差-方差权衡 | 实现复杂度 | 数据效率 |
|------|---------|-------------|-----------|---------|
| REINFORCE | 无 | 高偏差/高方差 | 简单 | 低（On-Policy） |
| A2C | 无 | 中偏差/中方差 | 简单 | 中 |
| TRPO | KL 散度约束 | 可调 | 复杂（需二阶优化） | 高 |
| **PPO-Clip** | **裁剪** | **可调（GAE）** | **简单** | **高** |
| PPO-Penalty | KL 惩罚 | 可调（GAE） | 中等 | 高 |

### 6. PPO 在 LLM-RLHF 中的变体

在大语言模型对齐场景中，PPO 有以下变化：

- **Actor**：被对齐的 LLM（策略模型）
- **Critic**：价值模型（通常与 Reward Model 共享底层）
- **Reference Model**：SFT 模型（用于计算 KL 散度惩罚）
- **Reward Model**：人类偏好训练的奖励模型
- **动作空间**：词表上的离散分布
- **状态**：当前的 token 序列
- **优势估计**：token-level 的 GAE

LLM-PPO 的额外损失项：

$$
L^{\text{KL}} = \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

用于防止策略偏离 SFT 模型太远，避免遗忘。

---

## 第二部分：代码实现

### 1. 环境准备与超参数

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym

# 超参数配置
class Config:
    # 网络结构
    state_dim = 8          # 状态维度（LunarLander-v2）
    action_dim = 4         # 动作维度
    hidden_dim = 64        # 隐藏层维度

    # PPO 核心参数
    gamma = 0.99           # 折扣因子
    lamda = 0.95           # GAE 参数 λ
    epsilon = 0.2          # 裁剪参数 ε
    K_epochs = 4           # 每次数据复用次数

    # 损失系数
    c1 = 0.5               # 价值损失系数
    c2 = 0.01              # 熵正则系数

    # 训练参数
    lr = 3e-4              # 学习率
    batch_size = 64        # 小批次大小
    max_episodes = 1000    # 最大回合数
    max_steps = 500        # 每回合最大步数
    update_freq = 2048     # 每多少步更新一次
```

### 2. Actor-Critic 网络

```python
class ActorCritic(nn.Module):
    """Actor-Critic 网络：共享底层，分别输出策略和价值"""

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor 头：输出动作概率分布
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic 头：输出状态价值
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        前向传播

        Args:
            state: 状态张量 [batch_size, state_dim]

        Returns:
            logits: 动作 logits [batch_size, action_dim]
            value: 状态价值 [batch_size, 1]
        """
        shared_features = self.shared(state)
        logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return logits, value

    def act(self, state):
        """
        选择动作

        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, state, action):
        """
        评估动作（用于更新阶段）

        Returns:
            log_prob: 动作的对数概率
            entropy: 策略的熵
            value: 状态价值
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value
```

### 3. GAE 优势估计实现

```python
def compute_gae(rewards, values, dones, dw, gamma=0.99, lamda=0.95):
    """
    计算广义优势估计（GAE）

    Args:
        rewards: 即时奖励列表 [T]
        values: 状态价值列表 [T+1]（包含最后一个 next_state 的价值）
        dones: 回合结束标志 [T]（广义结束，包括截断）
        dw: 回合真正结束标志 [T]（成功/失败）
        gamma: 折扣因子
        lamda: GAE 参数

    Returns:
        advantages: 优势估计 [T]
        returns: 目标价值 [T]
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0

    # 反向递推计算 GAE
    for t in reversed(range(T)):
        # TD Error: δ_t = r_t + γ * V(s_{t+1}) * (1 - dw) - V(s_t)
        delta = rewards[t] + gamma * values[t + 1] * (1 - dw[t]) - values[t]

        # A_t = δ_t + γ * λ * A_{t+1} * (1 - done)
        gae = delta + gamma * lamda * gae * (1 - dones[t])

        advantages[t] = gae

    # 目标价值 = 优势 + 当前价值估计
    returns = advantages + values[:T]

    return advantages, returns


# 测试
if __name__ == "__main__":
    # 模拟一个回合的数据
    T = 10
    rewards = torch.randn(T)
    values = torch.randn(T + 1)  # 多一个 next_state 的价值
    dones = torch.zeros(T)
    dones[-1] = 1  # 最后一步回合结束
    dw = torch.zeros(T)
    dw[-1] = 1

    advantages, returns = compute_gae(rewards, values, dones, dw)
    print("Advantages:", advantages)
    print("Returns:", returns)
```

### 4. PPO 核心损失函数实现

```python
def ppo_loss(model, states, actions, old_log_probs, advantages, returns,
             epsilon=0.2, c1=0.5, c2=0.01):
    """
    计算 PPO 损失函数

    Args:
        model: Actor-Critic 模型
        states: 状态批次 [batch_size, state_dim]
        actions: 动作批次 [batch_size]
        old_log_probs: 旧策略的对数概率 [batch_size]
        advantages: 优势估计 [batch_size]
        returns: 目标价值 [batch_size]
        epsilon: 裁剪参数
        c1: 价值损失系数
        c2: 熵正则系数

    Returns:
        total_loss: 总损失
        policy_loss: 策略损失（用于监控）
        value_loss: 价值损失（用于监控）
        entropy: 策略熵（用于监控）
    """
    # 获取新策略的评估结果
    new_log_probs, entropy, values = model.evaluate(states, actions)

    # ========== 1. 策略损失（Clipped Surrogate Objective） ==========

    # 重要性采样比率: r_t = exp(log π_θ - log π_θ_old)
    ratios = torch.exp(new_log_probs - old_log_probs)

    # 未裁剪的替代目标
    surr1 = ratios * advantages

    # 裁剪的替代目标
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages

    # 取最小值（保守更新），取负号因为要做梯度下降
    policy_loss = -torch.min(surr1, surr2).mean()

    # ========== 2. 价值函数损失 ==========

    # 均方误差（也可用 Smooth L1）
    value_loss = c1 * F.mse_loss(values.squeeze(), returns)

    # ========== 3. 熵正则化项 ==========

    # 取负号：损失中减去熵 = 最大化熵
    entropy_loss = -c2 * entropy.mean()

    # ========== 4. 总损失 ==========

    total_loss = policy_loss + value_loss + entropy_loss

    return total_loss, policy_loss.item(), value_loss.item(), entropy.mean().item()
```

### 5. 完整 PPO 训练流程

```python
class PPO:
    """PPO 算法完整实现"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化网络
        self.model = ActorCritic(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        # 经验缓冲区
        self.reset_buffer()

    def reset_buffer(self):
        """重置经验缓冲区"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.dw = []
        self.values = []

    def select_action(self, state):
        """
        选择动作

        Args:
            state: numpy 数组 [state_dim]

        Returns:
            action: 整数动作
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.model.act(state)

        # 存储经验
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value.squeeze())

        return action.item()

    def store_transition(self, reward, done, dw):
        """存储转移信息"""
        self.rewards.append(reward)
        self.dones.append(done)
        self.dw.append(dw)

    def update(self):
        """
        使用收集的经验更新网络

        Returns:
            avg_policy_loss: 平均策略损失
            avg_value_loss: 平均价值损失
            avg_entropy: 平均熵
        """
        # ========== 1. 计算 GAE ==========

        # 获取最后一个状态的价值
        with torch.no_grad():
            last_state = self.states[-1]
            _, last_value = self.model(last_state)
            self.values.append(last_value.squeeze())

        # 转换为张量
        rewards = torch.FloatTensor(self.rewards)
        values = torch.stack(self.values)
        dones = torch.FloatTensor(self.dones)
        dw = torch.FloatTensor(self.dw)

        # 计算 GAE
        advantages, returns = compute_gae(
            rewards, values, dones, dw,
            self.config.gamma, self.config.lamda
        )

        # 优势归一化（降低方差）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 准备训练数据
        old_states = torch.cat(self.states).to(self.device)
        old_actions = torch.stack(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)

        # ========== 2. 多轮更新 ==========

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0

        for _ in range(self.config.K_epochs):
            # 生成随机小批次索引
            indices = np.arange(len(self.rewards))
            np.random.shuffle(indices)

            for start in range(0, len(self.rewards), self.config.batch_size):
                end = start + self.config.batch_size
                batch_idx = indices[start:end]

                # 获取小批次数据
                batch_states = old_states[batch_idx]
                batch_actions = old_actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx].to(self.device)
                batch_returns = returns[batch_idx].to(self.device)

                # 计算损失
                loss, p_loss, v_loss, ent = ppo_loss(
                    self.model,
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns,
                    self.config.epsilon,
                    self.config.c1,
                    self.config.c2
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪（防止梯度爆炸）
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += p_loss
                total_value_loss += v_loss
                total_entropy += ent
                update_count += 1

        # 清空缓冲区
        self.reset_buffer()

        return (total_policy_loss / update_count,
                total_value_loss / update_count,
                total_entropy / update_count)
```

### 6. 完整训练循环

```python
def train():
    """PPO 训练主循环"""

    # 创建环境
    env = gym.make("LunarLander-v2")

    # 初始化
    config = Config()
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.n
    agent = PPO(config)

    # 训练记录
    episode_rewards = []
    total_steps = 0

    for episode in range(config.max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        dw = False
        step = 0

        while not done and step < config.max_steps:
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            dw = terminated  # 真正结束（成功/失败）

            # 存储转移
            agent.store_transition(reward, float(done), float(dw))

            state = next_state
            episode_reward += reward
            total_steps += 1
            step += 1

            # 按更新频率更新网络
            if total_steps % config.update_freq == 0:
                p_loss, v_loss, ent = agent.update()
                print(f"  Update at step {total_steps}: "
                      f"policy_loss={p_loss:.4f}, value_loss={v_loss:.4f}, entropy={ent:.4f}")

        episode_rewards.append(episode_reward)

        # 打印训练进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{config.max_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, Total Steps: {total_steps}")

    env.close()
    return episode_rewards


if __name__ == "__main__":
    rewards = train()
```

### 7. 数值稳定性技巧

```python
def ppo_loss_stable(model, states, actions, old_log_probs, advantages, returns,
                    epsilon=0.2, c1=0.5, c2=0.01):
    """
    数值稳定的 PPO 损失实现

    关键技巧：
    1. 使用 log 空间计算比率，避免数值溢出
    2. 优势归一化
    3. 梯度裁剪
    4. 使用 Smooth L1 替代 MSE
    """
    new_log_probs, entropy, values = model.evaluate(states, actions)

    # ========== 数值技巧 1: 在 log 空间计算比率 ==========
    # ratio = exp(new_log_prob - old_log_prob)
    # 这样比直接计算 new_prob / old_prob 更稳定
    log_ratios = new_log_probs - old_log_probs
    ratios = torch.exp(log_ratios)

    # ========== 数值技巧 2: 优势归一化（外层已做，这里可省略） ==========
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 策略损失
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # ========== 数值技巧 3: 使用 Smooth L1 替代 MSE ==========
    value_loss = c1 * F.smooth_l1_loss(values.squeeze(), returns)

    # 熵正则
    entropy_loss = -c2 * entropy.mean()

    total_loss = policy_loss + value_loss + entropy_loss

    return total_loss
```

### 8. 单元测试

```python
def test_gae():
    """测试 GAE 计算"""
    # 简单场景：3 步，最后一步结束
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 1.0, 1.5, 0.0])  # 最后一个是 next_state 价值
    dones = torch.tensor([0.0, 0.0, 1.0])
    dw = torch.tensor([0.0, 0.0, 1.0])

    advantages, returns = compute_gae(rewards, values, dones, dw, gamma=0.99, lamda=0.95)

    print("=== GAE 测试 ===")
    print(f"Advantages: {advantages}")
    print(f"Returns: {returns}")

    # 手动验证第一步
    # δ_2 = r_2 + γ * V(s_3) * (1 - dw_2) - V(s_2)
    #     = 3.0 + 0.99 * 0.0 * 0 - 1.5 = 1.5
    # A_2 = δ_2 = 1.5
    assert abs(advantages[2].item() - 1.5) < 1e-5, "最后一步优势计算错误"

    print("GAE 测试通过！")


def test_ppo_loss():
    """测试 PPO 损失计算"""
    model = ActorCritic(state_dim=4, action_dim=2, hidden_dim=32)

    batch_size = 8
    states = torch.randn(batch_size, 4)
    actions = torch.randint(0, 2, (batch_size,))
    old_log_probs = torch.randn(batch_size) - 1  # 模拟对数概率
    advantages = torch.randn(batch_size)
    returns = torch.randn(batch_size)

    loss, p_loss, v_loss, ent = ppo_loss(
        model, states, actions, old_log_probs, advantages, returns
    )

    print("=== PPO Loss 测试 ===")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Policy Loss: {p_loss:.4f}")
    print(f"Value Loss: {v_loss:.4f}")
    print(f"Entropy: {ent:.4f}")

    # 验证损失可以反向传播
    loss.backward()
    print("反向传播成功！")

    print("PPO Loss 测试通过！")


def test_clipping():
    """测试裁剪机制"""
    ratios = torch.tensor([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    advantages = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    epsilon = 0.2

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages

    print("=== 裁剪测试 ===")
    print(f"Ratios: {ratios}")
    print(f"Surr1 (未裁剪): {surr1}")
    print(f"Surr2 (裁剪后): {surr2}")
    print(f"Min: {torch.min(surr1, surr2)}")

    # 验证：ratio > 1.2 时被裁剪
    assert surr2[4].item() == 1.2, "ratio=1.5 应被裁剪到 1.2"
    assert surr2[5].item() == 1.2, "ratio=2.0 应被裁剪到 1.2"
    # ratio < 0.8 时被裁剪
    assert surr2[0].item() == 0.8, "ratio=0.5 应被裁剪到 0.8"

    print("裁剪测试通过！")


if __name__ == "__main__":
    test_gae()
    test_clipping()
    test_ppo_loss()
```

### 9. 面试手写版（精简版）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ========== 1. GAE 计算 ==========
def compute_gae(rewards, values, dones, dw, gamma=0.99, lamda=0.95):
    """
    rewards: [T] 即时奖励
    values: [T+1] 状态价值（含最后一个 next_state）
    dones: [T] 广义结束标志
    dw: [T] 真正结束标志
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] * (1 - dw[t]) - values[t]
        gae = delta + gamma * lamda * gae * (1 - dones[t])
        advantages[t] = gae

    returns = advantages + values[:T]
    return advantages, returns


# ========== 2. PPO 更新 ==========
def ppo_update(model, optimizer, states, actions, old_log_probs, advantages, returns,
               epsilon=0.2, c1=0.5, c2=0.01, K_epochs=4):
    """
    model: Actor-Critic 模型
    optimizer: 优化器
    states, actions: 采集的经验数据
    old_log_probs: 旧策略的对数概率
    advantages: GAE 优势估计
    returns: 目标价值
    """
    for _ in range(K_epochs):
        # 前向传播
        logits, values = model(states)
        dist = Categorical(logits=logits)

        # 新策略的对数概率和熵
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # 重要性采样比率
        ratios = torch.exp(new_log_probs - old_log_probs)

        # 策略损失（Clipped Surrogate）
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值损失
        value_loss = c1 * F.mse_loss(values.squeeze(), returns)

        # 总损失
        loss = policy_loss - c2 * entropy.mean() + value_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    return policy_loss.item(), value_loss.item()
```

---

## 总结

### PPO 算法完整流程

```
┌─────────────────────────────────────────────────────────┐
│                    PPO 训练流程                          │
├─────────────────────────────────────────────────────────┤
│  1. 用旧策略 π_θ_old 采集一批经验数据                     │
│     (states, actions, rewards, log_probs, values)        │
│                          ↓                               │
│  2. 计算 GAE 优势估计 A_t 和目标价值 V_target             │
│     - 反向递推计算 TD Error                              │
│     - 指数衰减求和得到优势                                │
│                          ↓                               │
│  3. 多轮更新（K epochs）                                  │
│     - 计算新策略比率 r_t = exp(log π_θ - log π_θ_old)    │
│     - Clipped Surrogate Loss                             │
│     - Value Function Loss                                │
│     - Entropy Bonus                                      │
│     - 梯度裁剪 + 参数更新                                 │
│                          ↓                               │
│  4. 重复 1-3 直到收敛                                     │
└─────────────────────────────────────────────────────────┘
```

### 核心公式速查表

| 组件 | 公式 |
|------|------|
| TD Error | $\delta_t = r_t + \gamma V(s_{t+1}) \cdot (1 - dw) - V(s_t)$ |
| GAE | $A_t = \delta_t + \gamma \lambda A_{t+1} \cdot (1 - done)$ |
| 比率 | $r_t(\theta) = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})$ |
| 策略损失 | $L^{\text{policy}} = -\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$ |
| 价值损失 | $L^{\text{value}} = (V_\phi(s_t) - V_t^{\text{target}})^2$ |
| 熵正则 | $L^{\text{entropy}} = -H(\pi_\theta(\cdot | s_t))$ |
| 总损失 | $L^{\text{total}} = L^{\text{policy}} + c_1 L^{\text{value}} - c_2 L^{\text{entropy}}$ |

### 关键超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| $\gamma$ | 0.99 | 折扣因子 |
| $\lambda$ | 0.95 | GAE 参数 |
| $\epsilon$ | 0.2 | 裁剪范围 |
| $c_1$ | 0.5 | 价值损失系数 |
| $c_2$ | 0.01 | 熵正则系数 |
| $K$ | 4 | 数据复用次数 |
| 学习率 | 3e-4 | Adam 优化器 |

---

## 参考资料

- OpenAI 原论文：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- GAE 原论文：[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- TRPO 原论文：[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- CleanRL PPO 实现：https://github.com/vwxyzjn/cleanrl
- PPO-simplest 教程：https://github.com/schinger/PPO-simplest
- 知乎：PPO损失函数详解
- 知乎：强化学习面试手撕--PPO
- Hwcoder：RL 学习笔记 #10 近端策略优化（PPO）理论
