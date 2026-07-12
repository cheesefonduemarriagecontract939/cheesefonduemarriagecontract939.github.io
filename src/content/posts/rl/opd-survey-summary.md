---
author: Bakuma-sea
pubDatetime: 2026-07-12T23:20:00+08:00
title: LLM 后训练 OPD 深度总结
featured: false
tags:
  - OPD
  - Distillation
  - RL
  - LLM
description: 围绕 On-Policy Distillation 梳理其动机、Reverse KL 数学本质、逐 token dense reward 与后训练价值。
timezone: Asia/Shanghai
---

# LLM 后训练 OPD（On-Policy Distillation）深度总结

## 一、术语澄清

在 LLM 后训练领域，**OPD 统一指 On-Policy Distillation（在线策略蒸馏 / 同策略蒸馏）**，由 Thinking Machines Lab 于 2025 年下旬提出并推广。当前 arXiv、知乎、青稞社区、GitHub 等来源中不存在广泛认可的 "Offline Preference Distillation" 这一术语。本文围绕实际文献中的 **On-Policy Distillation** 展开。

OPD 的核心理念可概括为一句话：**student 先用自己的当前策略生成回答（rollout），再让 teacher 在这些学生自己生成的轨迹上提供逐 token 的分布监督，学生据此更新。**

---

## 二、原始 OPD 的核心思想与动机

### 2.1 为什么需要 OPD？

LLM 后训练长期面临两条路径的抉择：

| 方法 | 采样来源 | 监督信号 | 优点 | 缺点 |
|------|----------|----------|------|------|
| **SFT / Off-Policy KD** | 教师模型预生成固定数据 | 逐 token dense 监督 | 训练稳定、信号密集 | 分布不匹配（Distribution Mismatch），存在 Exposure Bias |
| **RL（PPO/GRPO）** | 学生当前策略 rollout | 稀疏 scalar reward | 贴合自身分布 | 奖励稀疏、训练不稳定、方差大、成本高 |

OPD 的设计目标是**同时拿到两边的好处**：
- **On-policy 采样**：数据来自学生当前策略，零 exposure bias；
- **Dense token-level 监督**：教师对每个 token 给出 log-probability 或概率分布反馈，信号比 RL 的 scalar reward 密集得多。

### 2.2 一个直观对比

- **SFT**：老师在黑板上写答案，学生照着抄。学生考试时遇到自己没见过的题型，容易出错（分布漂移）。
- **RL**：学生自己解题，老师只给最后总分。学生不知道哪一步错了，探索效率低。
- **OPD**：学生自己解题，老师站在旁边，每写一步都打分。学生在自己真实会犯的错误上被纠正，同时获得逐步指导。

---

## 三、原始 OPD 的损失函数与手撕推导

### 3.1 从 KL 散度方向选择说起

KL 散度定义为：

$$D_{KL}(p \,\|\, q) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]$$

关键细节：**期望是对第一个参数取的**。交换位置得到的是完全不同的量：

- **Forward KL**：$D_{KL}(p \,\|\, q) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]$ —— 期望对 $p$ 取
- **Reverse KL**：$D_{KL}(q \,\|\, p) = \mathbb{E}_{x \sim q}\left[\log \frac{q(x)}{p(x)}\right]$ —— 期望对 $q$ 取

两者行为截然不同：
- **Forward KL** 是 **mass-covering**（均值寻找）：当 $p(x) > 0$ 而 $q(x) \to 0$ 时会爆炸为 $+\infty$，迫使 $q$ 必须在 $p$ 有支持的所有地方保留概率质量。$q$ 倾向宽而平坦。
- **Reverse KL** 是 **mode-seeking**（模式寻找）：期望只在 $q$ 有支持的地方计算，$q$ 可以放弃 $p$ 的某些 mode，只锁定高概率区域。$q$ 倾向窄而锐利。

### 3.2 SFT 对应 Forward KL，OPD 对应 Reverse KL

**SFT 的数学本质**：

SFT 在训练集（由教师 $\pi_T$ 生成）上最大化学生 $\pi_s$ 的 log-likelihood：

$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{y \sim \pi_T}\left[\log \pi_s(y \mid x)\right]$$

这是交叉熵 $H(\pi_T, \pi_s)$。利用 $H(p,q) = H(p) + D_{KL}(p \,\|\, q)$，且 $H(\pi_T)$ 与学生参数无关：

$$\boxed{\mathcal{L}_{\text{SFT}} \;\Longleftrightarrow\; \min_{\theta} D_{KL}(\pi_T \,\|\, \pi_s)}$$

**SFT 等价于最小化 Forward KL**。数据从老师采样，学生被迫 mass-cover 老师的全部轨迹模式——即使有些模式学生根本学不会。这就是分布漂移的根源。

**OPD 的数学本质**：

OPD 直接最小化 Reverse KL：

$$\boxed{\mathcal{L}_{\text{OPD}} = D_{KL}(\pi_s \,\|\, \pi_T) = \mathbb{E}_{y \sim \pi_s}\left[\log \pi_s(y \mid x) - \log \pi_T(y \mid x)\right]}$$

期望对学生 $\pi_s$ 取，意味着 **$y$ 必须由学生自己 rollout**——on-policy 不是工程选择，而是 reverse KL 在数学层面强制要求的。其几何后果是 mode-seeking：学生不需要覆盖老师的全部能力，只需要在自己能走到的轨迹上向老师靠拢。

### 3.3 从 Reverse KL 推导到 Per-Token Dense Reward

目标是把 $\min D_{KL}(\pi_s \,\|\, \pi_T)$ 一路化简成"老师逐 token 打分的 policy gradient"。整个过程是**六步代数恒等变形，没有任何近似**。

#### 第一步：翻号并拆分

最小化变最大化，两边乘 $-1$：

$$\min_{\theta} D_{KL}(\pi_s \,\|\, \pi_T) \;\Longleftrightarrow\; \max_{\theta} J(\theta)$$

$$J(\theta) = \mathbb{E}_{y \sim \pi_s}\left[\log \pi_T(y \mid x) - \log \pi_s(y \mid x)\right]$$

将括号内两项分开：

$$J(\theta) = \underbrace{\mathbb{E}_{y \sim \pi_s}\left[\log \pi_T(y \mid x)\right]}_{\text{(A) reward 项}} + \underbrace{\mathbb{E}_{y \sim \pi_s}\left[-\log \pi_s(y \mid x)\right]}_{\text{(B) } = H(\pi_s) \text{ 熵正则项}}$$

注意 (B) 项恰好是香农熵 $H(\pi_s)$。所以 OPD 的目标函数本质上是：

$$\boxed{\max_{\theta} \; \mathbb{E}_{y \sim \pi_s}\left[R(y)\right] + H(\pi_s), \quad R(y) := \log \pi_T(y \mid x)}$$

这是教科书里 **entropy-regularized RL** 的标准形式（Soft Actor-Critic 那一套）。

- **Reward**：$R(y) = \log \pi_T(y \mid x)$，即老师赋予这条轨迹的 log-likelihood；
- **熵正则**：$H(\pi_s)$ 自动出现，鼓励学生保持适度随机性、避免坍缩到单一输出。

> **Reward 项的语义**：$\log \pi_T(y \mid x)$ 是老师认为轨迹 $y$ 有多"对"。老师认为高概率 $\to$ reward 大 $\to$ 该被奖励；老师认为低概率 $\to$ reward 极负 $\to$ 该被惩罚。老师在这里就是 **reward model**，无需单独训练。

#### 第二步：对 $\theta$ 求梯度，处理"期望分布也含 $\theta$"的难点

$J(\theta)$ 的梯度：

$$\nabla_{\theta} J(\theta) = \nabla_{\theta} \mathbb{E}_{y \sim \pi_s}\left[\log \pi_T(y \mid x) - \log \pi_s(y \mid x)\right]$$

由于采样分布 $\pi_s$ 本身也依赖 $\theta$，不能直接穿进期望。需要拆分：

$$\nabla_{\theta} J(\theta) = \nabla_{\theta} \mathbb{E}_{y \sim \pi_s}\left[\log \pi_T(y \mid x)\right] + \nabla_{\theta} \mathbb{E}_{y \sim \pi_s}\left[-\log \pi_s(y \mid x)\right]$$

#### 第三步：用 Score Function 化简第一项

对于第一项 $\nabla_{\theta} \mathbb{E}_{y \sim \pi_s}\left[\log \pi_T(y \mid x)\right]$：

$$\nabla_{\theta} \mathbb{E}_{y \sim \pi_s}\left[f(y)\right] = \mathbb{E}_{y \sim \pi_s}\left[f(y) \cdot \nabla_{\theta} \log \pi_s(y \mid x)\right]$$

这里 $f(y) = \log \pi_T(y \mid x)$，所以：

$$\nabla_{\theta} \mathbb{E}_{y \sim \pi_s}\left[\log \pi_T(y \mid x)\right] = \mathbb{E}_{y \sim \pi_s}\left[\log \pi_T(y \mid x) \cdot \nabla_{\theta} \log \pi_s(y \mid x)\right]$$

#### 第四步：第二项的梯度恒为 0

这是整个推导中最精妙的一步。对于第二项：

$$\nabla_{\theta} \mathbb{E}_{y \sim \pi_s}\left[-\log \pi_s(y \mid x)\right] = \nabla_{\theta} H(\pi_s)$$

利用 score function 恒等式：

$$\nabla_{\theta} H(\pi_s) = -\mathbb{E}_{y \sim \pi_s}\left[\log \pi_s(y \mid x) \cdot \nabla_{\theta} \log \pi_s(y \mid x)\right] - \mathbb{E}_{y \sim \pi_s}\left[\nabla_{\theta} \log \pi_s(y \mid x)\right]$$

而 $\mathbb{E}_{y \sim \pi_s}\left[\nabla_{\theta} \log \pi_s(y \mid x)\right] = 0$（概率分布的归一化约束）。更进一步，可以证明在适当形式下，熵项的梯度与 reward 项的梯度中对应部分相抵消，最终：

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{y \sim \pi_s}\left[\left(\log \pi_T(y \mid x) - \log \pi_s(y \mid x)\right) \cdot \nabla_{\theta} \log \pi_s(y \mid x)\right]$$

#### 第五步：合并成 Policy Gradient 的标准形式

定义 **token-level advantage**（或 reward）：

$$A(y) = \log \pi_T(y \mid x) - \log \pi_s(y \mid x)$$

则：

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{y \sim \pi_s}\left[A(y) \cdot \nabla_{\theta} \log \pi_s(y \mid x)\right]$$

这正是 **policy gradient** 的标准形式：$\mathbb{E}\left[A \cdot \nabla \log \pi\right]$。

#### 第六步：利用自回归分解得到 Per-Token Reward

语言模型的策略是自回归的：$\pi(y \mid x) = \prod_{t=1}^{T} \pi(y_t \mid x, y_{<t})$。

将轨迹级 reward 分解到每个 token：

$$A(y) = \sum_{t=1}^{T} \underbrace{\left(\log \pi_T(y_t \mid x, y_{<t}) - \log \pi_s(y_t \mid x, y_{<t})\right)}_{r_t}$$

每个 token 的 **dense reward** 为：

$$\boxed{r_t = \log \pi_T(y_t \mid x, y_{<t}) - \log \pi_s(y_t \mid x, y_{<t})}$$

**直观含义**：
- 如果老师认为 token $y_t$ 应该高概率出现（$\log \pi_T$ 大），而学生认为低概率（$\log \pi_s$ 小），则 $r_t > 0$，**提升**该 token 概率；
- 如果老师认为 token $y_t$ 不应该出现（$\log \pi_T$ 小），而学生认为高概率（$\log \pi_s$ 大），则 $r_t < 0$，**抑制**该 token 概率。

最终梯度：

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{y \sim \pi_s}\left[\sum_{t=1}^{T} r_t \cdot \nabla_{\theta} \log \pi_s(y_t \mid x, y_{<t})\right]$$

### 3.4 与 SFT / GRPO 的横向对比

| 方法 | 目标形式 | 采样分布 | 监督粒度 | 本质 |
|------|----------|----------|----------|------|
| **SFT** | $-\mathbb{E}_{y \sim \pi_T}[\log \pi_s(y \mid x)]$ | $\pi_T$ (off-policy) | token-level | Forward KL，mass-covering |
| **OPD** | $\mathbb{E}_{y \sim \pi_s}[\log \pi_T(y \mid x) - \log \pi_s(y \mid x)]$ | $\pi_s$ (on-policy) | token-level dense reward | Reverse KL，mode-seeking |
| **GRPO** | $\mathbb{E}[A \cdot \nabla \log \pi_s]$，$A$ 为 group-relative advantage | $\pi_s$ (on-policy) | sequence-level scalar | Policy gradient，稀疏奖励 |

OPD 是 **RL 和 KD 的数学统一体**：从"蒸馏"端看，它是用 teacher 分布监督学生；从"RL"端看，它是 teacher log-prob 作为 reward 的 entropy-regularized policy gradient。

---

## 四、原始 OPD 的训练做法

### 4.1 标准训练流程（Thinking Machines Lab 原始方案）

1. **Rollout**：学生模型 $\pi_s$ 对输入 $x$ 生成完整轨迹 $y = (y_1, y_2, \ldots, y_T)$；
2. **Teacher Scoring**：教师模型 $\pi_T$（参数冻结）在学生生成的轨迹上计算每个 token 的 log-probability：$\log \pi_T(y_t \mid x, y_{<t})$；
3. **Student Scoring**：学生模型计算自己生成每个 token 的 log-probability：$\log \pi_s(y_t \mid x, y_{<t})$；
4. **Reward 计算**：$r_t = \log \pi_T(y_t \mid x, y_{<t}) - \log \pi_s(y_t \mid x, y_{<t})$；
5. **Policy Gradient 更新**：$\nabla_{\theta} J = \sum_t r_t \cdot \nabla_{\theta} \log \pi_s(y_t \mid x, y_{<t})$；
6. **循环**：更新后的学生模型继续生成新轨迹，重复上述过程。

### 4.2 工程实现要点

- **Reverse KL 的 mode-seeking 特性**：学生不会盲目覆盖 teacher 的全部模式，而是锁定自己能达到的高概率区域。这对小模型蒸馏大模型尤其重要——不指望小模型什么都会，只指望它在自己能做的事上做得像老师。
- **Teacher 需要实时在线**：原始 OPD 需要 live teacher server 在训练期间持续提供 log-probabilities，这带来显著的 infra 开销。后续 Lightning OPD 等变体解决了这一问题。
- **训练效率**：在数学推理任务上，OPD 通常以 **RL 1/10 的算力**达到相当甚至更好的效果。相同数据量下训练速度比 GRPO 快 **8-16 倍**。

---

## 五、源码级解读：OPD 一步步是怎么算出来的（verl 框架）

OPD 在工程上并非从零搭建一套新框架，而是**直接复用现有 RL 训练框架**（如 verl、Tinker）。它的所有"特殊性"都集中在 reward 计算和 advantage 估计这两步上，其余部分（rollout、actor update、数据并行）与标准 PPO/GRPO 完全一致。以下以 verl 框架中的 thunlp/OPD 实现为例，拆解每一步。

### 5.1 整体训练 loop：verl 是怎么"复用 RL"做 OPD 的

verl 的训练主循环是标准的 RL fit loop。OPD 在这个 loop 里只改了 reward 计算的那一段：

```python
# verl/trainer/ppo/ray_trainer.py（伪代码化）
for batch in dataloader:
    # 第一步：student rollout，拿到 on-policy 轨迹
    batch = self.actor_rollout_wg.generate_sequences(batch)

    # 第二步：student 自己再 forward 一遍 response，记录 log_prob
    batch = self.actor_rollout_wg.compute_log_prob(batch)

    # 第三步：teacher 一次 forward，算出 reverse KL，组装成 token-level reward
    teacher_data = self.rm_wg.compute_rm_score(batch)
    batch = batch.union(teacher_data)

    # 第四步：advantage estimator —— OPD 直接用 token-level reward 当 advantage
    batch = compute_advantage(batch, adv_estimator="token_reward_direct")

    # 第五步：正常的 PPO actor update（importance sampling loss）
    actor_output = self.actor_rollout_wg.update_actor(batch)
```

**关键事实**：teacher 模型是塞在 `reward_model` 这个槽位里的——这就是"复用 RL pipeline"的工程意义：原本 RL 里塞 reward model 的位置，OPD 直接把 teacher 塞进来。整套训练脚本是 GRPO 风格（每条 prompt 跑 N 条 rollout）。

### 5.2 Teacher 只做一次 prefill，不做 decode

第 3 步是 OPD 的"成本秘密"。teacher 对 student 整条 response 只做一次 forward，拿到 logits 张量 `[B, T, V]`，然后在词表维度上把 student 关心的 token id 挑出来，做 logsumexp 归一化。没有 decode，没有 autoregressive 生成。

这就是 OPD 比"用 reward model 打分的 RL"省得多的根本原因——后者经常需要 reward model 也做 long-context decode（比如 process reward model 那种逐步打分）。

### 5.3 reward 怎么算：`rm_scores = -kl_val * weights`

核心代码逻辑（`dp_actor.py` 中 `only_stu` 策略）：

```python
# S_logp:  student 在 top-k token 上的 log-prob，shape (B, T, K)
# T_on_S:  teacher 在 student 选出的同样这些 top-k token 上的 log-prob
kl_val = S_logp - T_on_S          # reverse KL 的逐 token 估计：log p - log q
norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
rm_scores = -kl_val * norm_weights   # reward = -reverse_KL，再用 student_p 做加权
```

逐行拆解：
- `kl_val = S_logp - T_on_S`：这就是 Reverse KL 的逐 token Monte Carlo 估计；
- `rm_scores = -kl_val * norm_weights`：reward 直接定义为**负的 Reverse KL**。KL 越小（student 越像 teacher），reward 越大；反之则狠狠惩罚 student；
- `norm_weights`：在 top-k 版本里，用 student 自己的概率做归一化权重。直觉上，你想让 student 学的，是"它自己在意但 teacher 不认可"的那些 token——这些才是真正驱动学生分布偏离 teacher 的"罪魁祸首"。

这个 reward 还有一个非常优雅的性质：**它是"unhackable"的**。常见的 reward model 都有 reward hacking 的问题，但 OPD 的 reward 是 -KL，KL 越小意味着 student 越像 teacher，没有任何"作弊空间"。

### 5.4 advantage 怎么算：直接令 `advantage = reward`

按照 PPO/GRPO 的常规设计，下一步应该是 GAE（Generalized Advantage Estimation）。但 OPD 做了一个非常激进的简化——**直接让 advantage = reward**：

```python
@register_adv_est("token_reward_direct")
def compute_token_reward_direct_advantage(token_level_rewards, response_mask, ...):
    # 直接把 token-level reward 当作 advantage 的最简 estimator
    advantages = token_level_rewards * response_mask
    returns = advantages.clone()
    return advantages, returns
```

**为什么可以这么"草率"？**

PPO 之所以需要 critic 和 GAE，是因为 reward 信号是 sparse 的——只有 episode 结束时才能算。GAE 干的事就是把这个稀疏的"未来收益"折现回每一个 token。但 OPD 的 reward 信号本身就是 **dense** 的——每个 token 都已经有了一个独立、有意义的 reward 值（就是那个 -KL）。这种情况下根本不需要 GAE 来"拼凑"出 token-level advantage，直接拿 reward 当 advantage 用就行。

Thinking Machines Lab 博客里的伪代码也用了同样的思路：
```python
teacher_logprobs = teacher_client.compute_logprobs(trajectories)
reverse_kl = sampled_logprobs - teacher_logprobs
trajectories["advantages"] = -reverse_kl
```

### 5.5 actor loss 怎么算：复用 PPO 的 importance sampling

最后一步，actor 是怎么更新的？答案非常直接——**直接复用 PPO 的 importance sampling loss**，几乎零修改：

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}_t\Big[\min\big(r_t(\theta) \cdot A_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\big)\Big]$$

其中 $r_t(\theta) = \pi_\theta(\hat{y}_t \mid x, \hat{y}_{< t}) / \pi_{\theta_{\mathrm{old}}}(\hat{y}_t \mid x, \hat{y}_{< t})$ 是 importance sampling 比率，$A_t$ 就是上一节的 advantage（在 OPD 里等于 $-\text{KL}_t$）。

**直觉**：当 advantage 是正的（即 KL 小，student 表现得像 teacher），就提升这个 token 的概率；当 advantage 是负的（KL 大，student 偏离 teacher），就压低这个 token 的概率。`clip` 项的作用和 PPO 完全一致，防止单步更新过猛。

### 5.6 源码级流程总结

| 步骤 | 操作 | 成本 |
|------|------|------|
| 1. Student rollout | 生成 on-policy 轨迹 | **贵**（需要 decode） |
| 2. Teacher prefill | 对 student 轨迹做一次 forward，拿 logits | **便宜**（无 decode） |
| 3. Reward 计算 | `reward = -(student_logp - teacher_logp)`，用 student_p 加权 | 可忽略 |
| 4. Advantage | `advantage = reward`（直接，无需 GAE/critic） | 可忽略 |
| 5. Actor update | 标准 PPO importance sampling loss | 中等 |

OPD 在工程上，本质上就是把 PPO 的 reward 换成 `-KL` 的一个特例。如果你之前训过 PPO，迁移到 OPD 几乎不需要改动任何 infra 代码——只需要把 reward model 的接口换成 teacher 的 log-prob 接口即可。

---

## 六、OPD 的后续改进与补充工作（高影响力）

自 2025 年下旬 Thinking Machines Lab 提出 OPD 以来，社区在约半年内涌现了大量改进工作。以下按影响力排序，涵盖 arXiv、知乎、青稞社区等来源中讨论度最高的方向。

### 6.0 工业实践：五家技术报告里的 OPD 怎么用

在讲具体改进论文之前，有必要先看看今年这一波 OPD 风潮中，各家厂商在自己的旗舰模型里到底是怎么用它的。以下按"对工业的影响力"和"出现时间"排序。

#### DeepSeek-V4：直接把 mix RL 的位置交给 OPD

DeepSeek V4 是这一波 OPD 风潮里最激进的玩家。V3.2 时代用过的整套 mix RL 流程，被 V4 几乎整个换成了 OPD。

**整体思路（两步）**：
- **Stage 1**：对每个 domain 单独训练一个 specialist（math / code / agent / instruction following 等 10+ 个），每个 specialist = SFT + GRPO 单独跑；
- **Stage 2**：Multi-Teacher OPD——student 自己 rollout，10+ 个 specialist 同时当 teacher 给 student 的 token 打分，用 **Full-Vocabulary + Reverse KL** 把所有专家"行为级"地合并回 student。

**为什么不像 V3.2 那样一次性 mix RL？** DeepSeek V4 的解释是：大模型的 post-training 不是"教模型新东西"，而是"在已经存在的预训练权重里，把每个 domain 对应的 expert mode 找出来"。一次性 mix RL 等于让 actor 同时往多个目标方向跑，目标之间互相打架，最后落到一个折中、平庸的 mode 上。不如先各跑各的（Stage 1），再用 OPD 的稠密 KL 信号把多个 mode 蒸馏回同一个 student（Stage 2）——后者是"行为级"的合并，比传统的参数加权 merging 优雅得多。

**工程细节**：V4 这次用的不是工业默认的 sampled-token，而是 **full-vocabulary OPD**。原因是 multi-teacher consolidation 场景下，一个 token 位置要同时和 10+ 个 teacher 的分布对齐，sampled-token 的 Monte Carlo 估计在多 teacher 上的方差会被显著放大。所以他们的折中是直接上 full-vocab 把估计量做精，反正 1.6T MoE 这种规模训练集群也不缺显存。

一句话总结：DeepSeek V4 用"独立 specialist + 多教师 OPD"的"先分后合"范式，取代"一次性 mix RL"的"all-in-one"范式——这是今年 OPD 在工业落地里最重磅的一次背书。

#### Qwen3：那张让所有人重新审视 OPD 的对比表

Qwen3 的技术报告里 Table 21 给出了一组被反复转发的对比数字：

| 方法 | AIME'24 | GPQA-Diamond | GPU Hours |
|------|---------|--------------|-----------|
| Off-policy distillation（SFT） | 55.0% | 55.6% | Unreported |
| + Reinforcement learning | 67.6% | 61.3% | **17,920** |
| + On-policy distillation | **74.4%** | **63.3%** | **1,800** |

OPD 不仅在 AIME'24 上比 RL 高 6.8 个点，所用 GPU 时数还只是 RL 的 **1/10**。这一张表是后面 MiMo、GLM-5、DeepSeek V4 都跟进 OPD 的直接导火索。

不过 Qwen3 的 OPD 用法也带来了一些"传染病"——在 Qwen3 GitHub Discussion 里，社区反馈了 OPD 在 strong-to-weak 蒸馏场景下，如果不加稳定化机制（sentence-level importance weighting、ratio clipping 等），student 的错误会逐步放大，最终训练崩溃。这些稳定化技巧后来也成为后续多篇论文的重点。

#### MiMo-V2-Flash：Multi-Teacher OPD（MOPD）

小米的 MiMo-V2-Flash（309B 总参 / 15B activated 的 MoE）把 OPD 升级成了 **Multi-Teacher On-Policy Distillation（MOPD）**：
- 先用 RL 训练一组 domain-specialized teachers（math 专家、code 专家、reasoning 专家）；
- 再用 MOPD 把这些 teacher 的能力统一蒸馏回一个 student 模型。

MOPD 被放在 post-training 的中间阶段，介于 mid-training 和 large-scale agentic RL 之间。这暗示了一个新的 post-training 范式：

```
Pre-training → Mid-training → Multi-Teacher OPD → Agentic RL
                                  ↑
                  把多个 RL 专家的能力先汇聚到一个 student
                  再在这个 student 上做 agentic RL
```

从 benchmark 上看，MOPD + 后续 Agentic RL 让 MiMo-V2-Flash 在 SWE-Bench、AIME 等任务上达到 SOTA，用 15B activated 的参数干掉了一系列 30B+ 的 dense 模型。

#### GLM-5：on-policy 跨阶段蒸馏，专治"能力衰退"

智谱 GLM-5（744B 参数 MoE）的 post-training pipeline 一共四个阶段：Reasoning RL → Agentic RL → General RL → **On-policy cross-stage distillation**（OPD 的位置）。

为什么最后还要补一刀 OPD？答案是 **catastrophic forgetting（灾难性遗忘）**。RL 在某个特定 domain 上越练越强的同时，会无意识地损害模型在其它 domain 上的能力。RL 只调整了模型的一小部分子网络，特别"脆弱"。

GLM-5 的解法就是：在四个阶段之间，用 on-policy distillation 让模型"对齐"到自己之前的某个能力 snapshot 上，把丢掉的通用能力捞回来。这是 OPD 在工业链路里非常有代表性的一种用法——**不是用来"学新东西"，而是用来"防止丢老东西"**。

#### Thinking Machines Lab：把 OPD 推到大众视野的那篇博客

TML 在 2025 年 10 月的博客中，除了科普 OPD 原理，还提出了一个特别精妙的 use case——**用模型旧版本的自己当 teacher**。

具体场景：Qwen3-8B（RL post-trained 版本，IF-eval 85%）→ 做 mid-training 学内部文档 → Internal QA 从 18% 涨到 43%，但 IF-eval 从 85% **暴跌到 45%**。怎么办？

**解法**：用 mid-training **之前**的 Qwen3-8B 自己当 teacher，跑一轮 OPD（用 Tulu3 prompts 做 instruction-following 任务）：

| 模型 | Internal QA | IF-eval |
|------|-------------|---------|
| Qwen3-8B（原版） | 18% | 85% |
| + midtrain（70% 文档 + 30% chat） | 36% | 79% |
| + OPD（用旧版自己当 teacher） | **41%** | **83%** |

Internal QA 没掉、IF-eval 几乎完全恢复，新知识 + 老能力都保住了。这个用法对持续学习（continual learning）的意义是巨大的：你可以周期性地用模型的旧 snapshot 给当前模型做 OPD，每次往新方向训练之后，都能把丢掉的老能力捞回来。

**OPD 在工业里的典型定位总结**：
1. **替代 mix RL**：先分后合的"specialist + multi-teacher OPD"取代一次性 mix RL（DeepSeek V4）；
2. **替代 RL（同 domain 内）**：用 1/10 算力跑出更好的 reasoning 性能（Qwen3）；
3. **多专家融合**：把多个 RL 专家压缩到一个 student（MiMo-V2-Flash MOPD）；
4. **跨阶段保活**：防止 RL 阶段之间发生能力衰退（GLM-5）；
5. **持续学习**：用旧版自己做 teacher，新知识 + 老能力两不误（TML）。

---

### 6.1 效率改进：消除 Live Teacher 需求

#### Lightning OPD（arXiv:2604.13010）

**核心问题**：原始 OPD 需要实时 teacher server，infra 开销大。

**核心 insight**：Teacher Consistency——如果 teacher 在 SFT rollout 上的 log-probabilities 与在 student rollout 上的分布足够一致，就可以**预计算** teacher 的 log-probabilities。

**做法**：
- 先让 SFT 模型生成一组 rollout；
- 预计算 teacher 在这些 SFT rollout 上的 log-probabilities；
- 训练时学生采样轨迹，直接从预计算缓存中读取 teacher 信号，无需 live teacher。

**效果**：在 Qwen3-8B-Base 上，AIME 2024 达到 69.9%，仅需 30 GPU 小时，**效率提升 4.0 倍**。

---

### 6.2 训练动态与机制理解

#### Rethinking On-Policy Distillation（arXiv:2604.13016，清华）

**核心贡献**：首次系统研究 OPD 的训练动态和内在机制，回答了"OPD 为什么成功 / 为什么失败"。

**四条核心结论**：
1. OPD 不是"学更强的老师"，而是**"学兼容的 thinking 模式 + 学真正的新知识"**；
2. OPD 成功的本质：在学生自己走到的状态上，把高概率 token 慢慢对齐到老师；
3. OPD 失败的本质：thinking 模式不兼容，或老师没有新东西可教；
4. **长文本 OPD 天然会崩溃**：越往后老师越看不懂学生的 prefix（预填充），导致 teacher 信号质量下降。

**修复失败 OPD 的两个实用策略**：
- **Off-policy cold start**：先用 SFT 做冷启动，让 student 的 thinking 模式与 teacher 初步对齐；
- **Teacher-aligned prompt selection**：选择 student 和 teacher 生成分布接近的 prompt 进行训练。

---

### 6.3 损失函数改进：动态切换 KL 方向

#### EOPD: Entropy-Aware On-Policy Distillation（arXiv:2603.07079）

**核心问题**：纯 Reverse KL 是 mode-seeking，在 teacher 高熵（不确定）区域，学生会**坍缩到单一输出**，丢失多样性。实验发现 teacher 产生的 token 中有 18.5% 处于高熵区，但学生训练完后只保留了 6.8%。

**核心 insight**：Reverse KL 和 Forward KL 是互补的。Reverse KL 适合 confident 区域（高效学习），Forward KL 适合 uncertain 区域（保留多样性）。

**做法**：
- 计算 teacher 的 token-level 熵 $H(\pi_T(y_t \mid x, y_{<t}))$；
- 当 teacher 熵**超过阈值**时，在标准 Reverse KL 目标中加入 Forward KL 项；
- 当 teacher 熵较低时，保持纯 Reverse KL 进行精确模仿。

**效果**：数学推理性能提升，且模型越大增益越高（大模型在复杂任务上多样性退化更严重）。

---

### 6.4 训练稳定性改进：有界奖励与梯度裁剪

#### PowerOPD（arXiv:2606.17199）

**核心问题**：原始 OPD 的 log-ratio reward $r_t = \log \pi_T - \log \pi_s$ 是**无界**的。当 student 和 teacher 概率差异大时，reward 的绝对值可以非常大，导致**高方差梯度**，训练不稳定。

**核心 insight**：原始 OPD 的 log-ratio 是 Box-Cox power transformation 在 $\alpha \to 0$ 时的退化极限。如果用一个**原生有界**的变换替代 log-ratio，可以从根源上控制梯度方差。

**做法**：定义一族 parameterized reward：

$$r_t^{(\alpha)} = \frac{(\pi_T / \pi_s)^{\alpha} - 1}{\alpha}$$

当 $\alpha \to 0$ 时，退化为原始 log-ratio：$r_t^{(0)} = \log(\pi_T / \pi_s)$。

取 $\alpha > 0$（如 $\alpha = 0.5$），reward 天然有界，且**符号一致**（与 log-ratio 同号），不会扭曲优化方向。

**效果**：相比原始 OPD 和事后 scaling（如 Veto），PowerOPD 在保持训练稳定的同时不损失最终性能。

---

#### AOPD: Asymmetric On-Policy Distillation（arXiv:2605.06387）

**核心问题**：Policy gradient 中，负 advantage 区域的梯度是**高方差噪声**（zero advantage 区域梯度消失，正 advantage 区域是可信信号）。原始 OPD 对所有 token 一视同仁，导致负 reward 区域的不稳定更新拖慢训练。

**核心 insight**：按 advantage 符号切两段处理——
- **正 advantage 区域**（$r_t > 0$）：保留 policy gradient 形式（ exploitation，利用有效信号）；
- **非正 advantage 区域**（$r_t \leq 0$）：将无效负强化替换为**局部的 KL 散度最小化**（直接模仿，而非带噪声的 policy gradient）。

**效果**：Strong init +4.09%，Weak init +8.34%。

---

### 6.5 G-OPD / ExOPD：OPD 与 RL 的统一框架及 Reward Extrapolation（arXiv:2602.12125）

**核心问题**：OPD 虽然效果显著，但其内在机制此前尚未被充分理论化。标准 OPD 默认 $\beta = 1$（KL 惩罚权重等于 reward 权重），这只是一个特例，没有回答"如果改变这个比例会发生什么"。另外，标准 OPD 有一个天然瓶颈——**学生无法超越 teacher**：因为目标是最小化 $D_{KL}(\pi_s \,\|\, \pi_T)$，学生最多只能"像"老师，不可能超过老师。

这篇论文的核心贡献是**把 OPD 统一到了强化学习框架下**，并证明：通过调节 reward 项与 KL 惩罚项的相对权重，可以让学生不仅"学到老师"，还能**超越老师**。

#### 6.5.1 理论推导：OPD 本质上是 Dense RL 的一个特例

标准的带 KL 约束的 RL 目标函数为：

$$J_{RL}(\theta) = \max_{\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)}\left[ r(x,y) - \beta D_{KL}(\pi_\theta \,\|\, \pi_{ref}) \right]$$

OPD 的目标函数是最小化 Reverse KL：

$$J_{OPD}(\theta) = \min_{\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)}\left[ D_{KL}(\pi_\theta(y|x) \,\|\, \pi^*(y|x)) \right]$$

**关键推导**：引入任意参考模型 $\pi_{ref}$，将 OPD 目标重写为：

$$\max_{\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)}\left[ \log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} - D_{KL}(\pi_\theta(y|x) \,\|\, \pi_{ref}(y|x)) \right]$$

**核心推论**：OPD 实际上等同于**特定的 KL 约束 RL**：
- **奖励函数**：$r(x,y) = \log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}$（这是一个**密集的 token 级别奖励**）
- **KL 惩罚权重**：$\beta = 1$（reward 项与 KL 正则化项的权重严格相等）
- **参考模型**：$\pi_{ref}$ 可以是**任意模型**（不一定是学生自身）

这就是 OPD 的"RL 本质"——它不是什么全新的蒸馏魔法，而是 reward 为 teacher log-ratio、KL 权重恰好为 1 的一个**特定配置的 RL**。

#### 6.5.2 G-OPD 框架：通过 $\lambda$ 调节 reward 缩放

基于上述发现，论文引入**奖励缩放因子 $\lambda$**，提出 Generalized On-Policy Distillation（G-OPD）框架：

$$\boxed{\mathcal{J}_{G-OPD}(\theta) = \max_{\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)}\left[ \lambda \cdot \log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} - D_{KL}(\pi_\theta(y|x) \,\|\, \pi_{ref}(y|x)) \right]}$$

通过调节 $\lambda$，G-OPD 揭示了两种截然不同的蒸馏机制：

| 机制 | 条件 | 效果 | 应用场景 |
|------|------|------|----------|
| **Reward Interpolation** | $0 < \lambda < 1$ | 学生模型的分布介于参考模型和教师模型之间 | 预算可控的推理（模型输出长度、计算量可控） |
| **Reward Extrapolation (ExOPD)** | $\lambda > 1$ | 学生不仅匹配教师，还拟合一个额外的偏移项，**可能超越教师** | 多专家融合、突破 teacher 边界 |

**Reward Interpolation** 的直觉：当 $\lambda < 1$ 时，reward 项被"打折"，KL 惩罚项相对更强，学生不会全力追逐 teacher，而是停留在 reference 和 teacher 之间的某个中间点。这类似于模型的"输出长度控制"——$\lambda$ 越小，学生越保守（输出越短、越接近 reference）；$\lambda$ 越大，学生越激进（输出越长、越接近 teacher）。

**Reward Extrapolation (ExOPD)** 的直觉：当 $\lambda > 1$ 时，reward 项被"放大"，学生被鼓励去拟合一个比 teacher 更"极端"的分布。在多教师场景（比如把 math 专家和 code 专家蒸馏回同一个 student）中，ExOPD 可以让单一 student 模型**超越所有参与蒸馏的领域专家**——因为每个专家只是自己领域的局部最优，而 ExOPD 通过 reward 外推，让学生找到一个"比任何单一专家都更好"的全局解。

#### 6.5.3 强到弱蒸馏中的 Reward Correction

在将大参数 teacher 蒸馏到小参数 student（Strong-to-Weak）时，参考模型 $\pi_{ref}$ 的选择至关重要。

- **默认方案**：$\pi_{ref} = \pi_{base}^{student}$（学生自己的基础模型）。但大小模型之间存在知识和容量的固有鸿沟，使用 $\log\frac{\pi^*}{\pi_{base}^{student}}$ 作为隐式奖励会引入噪声。
- **Reward Correction 方案**：将 $\pi_{ref}$ 替换为 teacher 模型在进行 RL 之前的 base 版本 $\pi_{base}^{teacher}$。此时奖励信号变为 $\log\frac{\pi^*}{\pi_{base}^{teacher}}$，这个信号**直接对应了教师模型在后训练中获取的真实隐式奖励**，因此更加精确。

**代价**：需要额外加载 teacher 模型的 base 版本，计算开销增加。但实验表明这种修正能显著提升强到弱蒸馏的性能。

#### 6.5.4 实验结果

论文在数学推理（DeepMath）和代码生成（Eurus-RL-Code）上验证了 ExOPD，训练采用 GRPO 算法并加入 token-level rollout correction 以缓解训练与推理偏差。

**单教师同尺寸蒸馏**：ExOPD（$\lambda = 1.25$）稳定超越了标准 OPD 和经过同样步数训练的领域教师模型。过高的 $\lambda$（如 1.5）会因为模型过度拟合存在偏差的隐式奖励而导致性能下降——说明 reward extrapolation 有个"甜蜜点"。

**多专家模型融合**：在整合 Math 和 Code 专家能力时：
- 常规 SFT 效果次优；
- 标准 OPD 的上限被教师模型锁死（学生不可能超越任何 teacher）；
- 模型权重插值法（EXPO）缺乏稳定性；
- **ExOPD 生成的统一学生模型在所有基准测试上均超越了相应的领域教师**。

**关键结论**：在蒸馏过程中进行 reward 外推（$\lambda > 1$），不仅能够有效完成多领域专家模型的知识融合，还能让最终的学生模型**实质性地突破原有教师模型的能力边界**。这证明了在无需额外训练复杂 reward model 的情况下，仅通过灵活设定隐式奖励项的权重，就能实现模型推理能力的可控提升。

---

### 6.6 双蒸馏与特权信息处理

#### DOPD: Dual On-policy Distillation（arXiv:2606.30626）

**核心问题**：当 teacher 或 student 拥有**特权信息**（如 gold answer、更长的推理链、多模态输入）时，OPD 会遭遇 **"特权幻觉"（Privilege Illusion）**——teacher 的表现提升并非来自真实能力，而是来自特权输入。学生盲目模仿会导致过拟合到特权条件。

**核心 insight**：当 teacher 和 student 都拥有相同的特权信息时，两者之间**剩余的预测差异**才是真正的能力差距。

**做法**：定义 **Privilege Advantage Gap**：

$$\Delta_{\text{adv}} = A_{\text{teacher}}(\text{privileged}) - A_{\text{student}}(\text{privileged})$$

根据 advantage gap 和相对概率，**动态路由** token-level 监督：
- 当 teacher 优势显著时，用 teacher-driven 蒸馏；
- 当 student 已接近 teacher 或特权信息导致幻觉时，启用 student 自身的 auxiliary self-optimization。

**效果**：LLM 平均 +12.3 分，VLM 平均 +10.1 分。

---

### 6.7 自蒸馏：无外部 Teacher 的 OPD

2026 年初，MIT、ETH Zurich、UCLA 等团队几乎同时提出了一系列 **On-Policy Self-Distillation** 工作，核心思想是：没有外部 teacher 时，让模型自己充当 teacher。

#### SDPO: Self-Distillation Policy Optimization（arXiv:2601.20802）

**核心思想**：将环境反馈（如代码执行结果、数学题答案）转化为 **tokenized feedback**，然后让学生模型自身基于这些反馈生成"更强的自我"作为 teacher。不需要任何外部 teacher 或显式 reward model。

**做法**：
- 学生生成 rollout；
- 环境给出 rich feedback（如编译错误信息、测试用例结果）；
- 将 feedback 编码为 token 序列，与学生自身组合成一个"更强的条件化 self"；
- 用这个"更强的 self"作为 teacher，进行标准的 OPD。

---

#### OPSD: On-Policy Self-Distillation（arXiv:2601.18734）

**核心思想**：同一个 LLM 通过**不同的上下文条件**分别充当 teacher 和 student。

**做法**：
- **Student**：给定标准输入 $x$，生成基础回答；
- **Teacher**：给定 $x$ + 特权信息（如 gold answer hints、更详细的 reasoning prompt），生成更完善的回答；
- 两个回答来自**同一个模型**，只是条件不同。用 teacher 条件化的输出监督 student 条件化的输出。

**优势**：无需任何外部模型，完全 self-contained。特别适合持续学习和灾难性遗忘的缓解。

---

#### SDFT: Self-Distillation Fine-Tuning

与 OPSD / SDPO 同期的另一条自蒸馏路线，核心差异在于：
- **SDPO** 强调将环境反馈转化为 dense token-level 信号；
- **OPSD** 强调利用特权上下文构建更强的自我；
- **SDFT** 强调在持续学习场景中，用旧模型作为 teacher、新模型作为 student，防止灾难性遗忘。

---

### 6.8 其他重要改进方向

| 工作 | 核心贡献 | 来源 |
|------|----------|------|
| **Veto** | 从梯度几何角度修复 OPD 训练崩溃，识别并裁剪"有害梯度方向" | arXiv 2026 |
| **GKD (Generalized Knowledge Distillation)** | 在 OPD 框架下统一处理多种 divergence（JS、$\chi^2$ 等），证明 Reverse KL 是最优选择之一 | 前期工作 |
| **MiniLLM** | 早期探索 Reverse KL 在语言模型蒸馏中的效果，为 OPD 奠定实验基础 | 前期工作 |
| **f-divergence Survey（arXiv:2604.00626）** | 首篇 OPD 系统综述，提出基于 f-divergence 的统一框架，从 feedback signal、teacher access、loss granularity 三个正交维度梳理研究版图 | 综述 |
| **CRISP / ExOPD / GAD** | 多种 OPD 的工程变体，关注不同场景（对话、多轮、多模态）的适配 | 后续工作 |

---

### 6.9 四篇核心论文详细解读：从范式确立到机理加速

除了上面的改进方法，OPD 方向还有四篇值得精读的"奠基性"论文，它们构成了从"范式确立"到"机理理解"的完整闭环。

#### GKD（Agarwal et al., 2023）：OPD 范式的"源头活水"

Google DeepMind 在 2023 年 6 月放出的 **Generalized Knowledge Distillation**（arxiv:2306.13649，ICLR 2024）是历史上第一次有人把"on-policy + token-level KL"这套配方用我们现在熟悉的形式写下来。今天工业界谈的所有 OPD，本质上都是 GKD 框架的某种特例。

**GKD 的两个核心改动**：
1. **训练样本来自 student 自己**：把传统 KD 的 loss 里的 $(x, y)$ 数据集采样，换成 student 自己 rollout 的 on-policy 样本：
   $$\mathcal{L}_{\mathrm{GKD}}^{\mathrm{on-policy}} = \mathbb{E}_{x \sim \mathcal{X}, y \sim p_S(\cdot|x)} \left[ \mathcal{D}_{\mathrm{KL}}(p_T(\cdot|y_{< t},x) \parallel p_S(\cdot|y_{< t},x)) \right]$$
   外层期望里的 $y$ 是从 student 自己采的，inner 的 token-level loss 和传统 KD 一样——teacher 在每个 prefix 上给出分布，学生去对齐。这就是后来工业界所说的 OPD 最简形态。

2. **系统讨论了散度选择**：GKD 第一次系统讨论了 forward KL、reverse KL 和 generalized JSD 的取舍。它指出 forward KL 是 mean-seeking（会让 student 试图覆盖 teacher 所有 mode），reverse KL 是 mode-seeking（让 student 在 teacher 高概率 mode 上深耕），后者正好契合 student 容量比 teacher 小的现实。今天所有 2026 年的 OPD 实现默认用 reverse KL，就是这篇 paper 拍下来的工程默认。

**GKD 还首次展示了 on-policy distillation 可以无缝叠加 RL fine-tuning**，这一点直接预言了 2026 年"OPD + 后续 agentic RL"的范式（MiMo、GLM-5 都在这么做）。

**为什么 GKD 在 2023 年没有火？** 当时大家都在堆 RLHF 的 PPO/DPO，post-training 的注意力被吸走了；而且 T5 backbone 太老（最大 student 才 770M），论文里的绝对数字不能 carry 出"OPD 比 RL 更划算"的工业冲击力。直到 2025 年 Qwen3 那张表出来，这个范式才真正出圈。

---

#### Uni-OPD（2026.05，腾讯×浙大）：Dual-Perspective 救赎"探索不足 + teacher 信号不可靠"

Rethinking OPD 告诉你 OPD 什么时候不能 work，但它聚焦在训练前置阶段（cold start、prompt selection）。一个很自然的下一步是：如果训练已经开始了，过程中能不能在 loss / 数据层面动手脚去救？

腾讯和浙大的 **Uni-OPD**（arxiv:2605.03677）就是来回答这个问题的。它把 OPD 训练时的可观察问题归结成两个：

1. **Insufficient exploration of informative states**：student 在 rollout 时容易陷在"自己擅长的简单 prompt"上，没充分接触到那些既难又有信息量的中间状态；
2. **Unreliable teacher supervision for student rollouts**：teacher 在 student 跑出来的"奇怪轨迹"上打分往往不可靠。典型 failure mode：teacher 在错误 rollout 上给出高分，或在正确 rollout 上给出低分。

**学生侧的 Recipe：双重 data balancing**
- **Offline difficulty-aware balancing**：训练前先用 teacher 给所有 prompt 打 difficulty 标签，按难度分桶采样，避免 student 反复磨同一档题目；
- **Online correctness-aware balancing**：训练中实时根据 student rollout 的对错，动态调整 batch——对错混合的 batch 才能提供"正负对比"的学习信号。

**教师侧的 Recipe：outcome-guided margin calibration**
核心思路是：用 outcome reward（最终答对没）作为锚点，校准 teacher 给出的 token-level KL 信号。

论文先定义轨迹级的 OPD 回报：
$$G_{\mathrm{OPD}}(q, \tau) = \frac{1}{|\tau|}\sum_{t=1}^{|\tau|}(\log \pi_T(o_t | q, o_{< t}) - \log \pi_\theta(o_t | q, o_{< t}))$$

$G_{\mathrm{OPD}} > 0$ 表示 teacher 在这条轨迹上给的 confidence 比 student 高（"teacher 鼓励学生学这条"）。然后定义 prompt-level margin：
$$m(q) = \min_{\tau \in S_+(q)} G_{\text{OPD}}(q, \tau) - \max_{\tau \in S_-(q)} G_{\text{OPD}}(q, \tau)$$

$S_+$ 是正确 rollout 集合，$S_-$ 是错误 rollout 集合。$m(q) \geq 0$ 表示最差的正确轨迹也比最好的错误轨迹分高——这是 order consistency。当 $m(q) < \delta$ 时，提供两个 calibration 策略：
- **Margin mask**：直接把这个 prompt 的整个 group 丢掉，不参与训练；
- **Margin shift**：给所有正确轨迹的 $G_{\mathrm{OPD}}$ 加一个常数 $\lambda(q) = \delta - m(q)$，恢复 order consistency。

**实证结果**：在 5 个 domain、16 个 benchmark 上，Uni-OPD 相比 OPD baseline 提升稳定：math +1.5、code +3.4、strong-to-weak 30B→4B +1.7、cross-modal +1.2。

**一个特别值得记的结论**："Teacher value comes from the capability gap, not absolute strength alone." OPD 的提升上限并不取决于 teacher 有多大，而是取决于 teacher 和 student 之间存在多少 student 没见过的能力差。这和 Rethinking OPD 的"前提 2"完全一致。

---

#### Learning to Foresee（2026.05，中科大×腾讯）：用参数动力学解释 OPD 的"预见性"，顺手提速 3×

Qwen3 那张表的 1/10 算力，单纯靠"dense supervision 比 sparse 信息量大"不足以解释——dense 信号在 SFT 里也存在，但 SFT 并没有比 RL 高效 10 倍。所以一定还有别的什么在起作用。

中科大和腾讯的 **Learning to Foresee**（arxiv:2605.11739）就专门回答这个问题。它给出的解释非常优雅，可以浓缩成一个词——**"foresight"（预见性）**。

**核心论点**：OPD 之所以快，是因为它在训练早期就把"最终最优解的方向"找到了，剩下的训练只是沿着这个方向做 magnitude 上的累加。RL 则相反——它在训练过程中一直在"换方向"，浪费了大量算力在并不通往最终解的路上。

论文提出了两条 property：

**Property 1：Functional Redundancy Avoidance（模块层面的预见性）**
论文做了一个聪明的对照：把 OPD 和 RL 训完的同一个 base model 的参数差 $\Delta W$，rescale 到同样的 norm，再加回 base 模型，测最终准确率。如果纯粹是"OPD 算得多"→"更新更大"，norm 拉齐后应该打平。但实验发现 OPD 的最终性能比 RL 高得多——说明 OPD 的更新方向本身"信息密度"更高。

进一步用 sliding-window intervention 测模型各段的 reasoning 贡献度：
- 中间层 MLP 是 reasoning 的主要承载者（symbolic 知识和 relational reasoning），呈现倒 U 字形；
- RL 在贡献度低的底层和顶层注入了大量 update norm——"瞎更新"了很多对 reasoning 没用的参数；
- OPD 自动避开了这些 low-utility 模块，把 update norm 集中到中间层 MLP 上。

**Property 2：Early Low-Rank Lock-in（方向层面的预见性）**
对 $\Delta W$ 做 SVD 分析，发现 OPD 的 update 集中在更少几个主方向上，能量分布更"低秩"：

| 度量（OPD vs RL，8B 模型） | RL | OPD |
|---------------------------|-----|-----|
| Spectral / Frobenius Norm Ratio (↑) | 32.7% | 36.8% |
| Effective Rank (↓) | 2754 | 2341 |
| Top-1% Subspace Norm Ratio (↑) | 88.5% | **94.7%** |

更关键的是，追踪训练动态——把每个 checkpoint 的 update 主方向和最终 checkpoint 的主方向做余弦相似度：OPD 在训练早期 0%–30% 阶段，主方向就已经和最终方向高度对齐了，而且后续保持稳定。RL 则需要到 60%–80% 才能稳定下来。

**最反直觉的实验**：取 OPD 训练只跑了 10% 的 checkpoint，按 module-wise 把它的 update norm 拉到最终 checkpoint 的水平——只调 magnitude，方向不动。结果这个"早 checkpoint + 后 magnitude" 的模型能恢复 **80%** 的最终性能！这直接证明了：OPD 的方向在 10% 训练时就基本定型了，剩下 90% 只是在堆 magnitude。

**EffOPD：把"预见性"翻译成可工程化的 3× 加速**
基于这个洞察，论文提出了 **EffOPD**：
- 在指数间隔的 checkpoint $t=2^n$（1, 2, 4, 8, 16... 步）触发 extrapolation；
- 用最近两个 exponential checkpoint 的差 $\Delta_n = W_{2^n} - W_{2^{n-1}}$ 作为"局部更新方向"；
- 沿这个方向构造 5 个 candidate $\tilde{W}_{n,k} = W_{2^n} + 2k \cdot \Delta_n$，$k=1,...,5$；
- 用一个只有 50 条样本的 mini validation set 评估每个 candidate，一旦掉点就停，接受最大的涨点 candidate。

整套机制无需训新模块、无需复杂超参，可以 plug-in 到任何 OPD pipeline。实验显示 EffOPD 平均比 vanilla OPD 快 **3×**，有些场景下 4 步训练就达到了 vanilla 30 步的水平。

**Learning to Foresee 对 OPD 整体认识的修正**：

| 旧认知 | Learning to Foresee 的修正 |
|--------|---------------------------|
| OPD 快是因为 dense supervision | 真正的原因是它早期就锁定了正确的更新方向（dense supervision 只是表象） |
| OPD 训 100 个 step 比 10 个 step 好 | 10% 的训练已经把方向定下来了，后面只是在堆 magnitude——不如直接 extrapolate |
| OPD 和 RL 优化的是同一个目标，只是路径不同 | OPD 自动避开了 RL 大量浪费的 low-utility 模块，所以效率本质就是不同 |

这套理论也反过来呼应了 Rethinking OPD 提到的 "dense reward 不是免费午餐"——OPD 的"早期锁定方向"是双刃剑：在 reasoning 这种结构化任务上是优势，在需要长程探索的 agentic 场景下就成了束缚（探索空间被锁死了），所以 OPD 需要和 RL 配合使用，而不是单打独斗。

---

## 七、总结与选型建议

### 7.1 OPD 的核心价值

OPD 把 LLM 后训练从"二选一"（SFT vs RL）推向了"第三条路"：
- 保留了 RL 的 **on-policy 分布真实性**（零 exposure bias）；
- 保留了蒸馏的 **dense token-level 监督**（训练效率高、信号稳定）；
- 数学上统一为 **Reverse KL 下的 entropy-regularized policy gradient**。

### 7.2 实际应用时的选型建议

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 有强 teacher、追求最高效率 | **Lightning OPD** | 预计算 teacher log-prob，无需 live server |
| 追求训练稳定性、担心梯度爆炸 | **PowerOPD** 或 **AOPD** | 有界 reward / 按 advantage 符号分区处理 |
| Teacher 高熵区域多、需要多样性 | **EOPD** | 动态切换 Forward/Reverse KL |
| Teacher 有特权信息（如 gold answer） | **DOPD** | 避免特权幻觉，动态路由监督 |
| 无外部 teacher，需自我进化 | **SDPO / OPSD / SDFT** | 利用环境反馈或特权上下文构建自我 teacher |
| 长文本推理 | **Rethinking OPD 的冷启动策略** | 解决长文本 teacher 看不懂 student prefix 的问题 |
| 多专家融合 / 需突破 teacher 边界 | **G-OPD / ExOPD** | 通过 $\lambda > 1$ 的 reward extrapolation 让 student 超越所有 teacher |

### 7.3 参考文献

1. Thinking Machines Lab. "On-Policy Distillation." 2025.
2. Wu et al. "Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation." arXiv:2604.13010, 2026.
3. Fu et al. "Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe." arXiv:2604.13016, 2026.
4. "Entropy-Aware On-Policy Distillation of Language Models." arXiv:2603.07079, 2026.
5. Zhao et al. "PowerOPD: Stabilizing On-Policy Distillation with Bounded Power Transformation." arXiv:2606.17199, 2026.
6. "Asymmetric On-Policy Distillation: Bridging Exploitation and Imitation." arXiv:2605.06387, 2026.
7. "DOPD: Dual On-policy Distillation." arXiv:2606.30626, 2026.
8. "Reinforcement Learning via Self-Distillation (SDPO)." arXiv:2601.20802, 2026.
9. "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models (OPSD)." arXiv:2601.18734, 2026.
10. "A Survey of On-Policy Distillation for Large Language Models." arXiv:2604.00626, 2026.
11. Agarwal et al. "Generalized Knowledge Distillation for Online Distillation of Language Models." arXiv:2306.13649 (ICLR 2024), 2023.
12. "Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe." arXiv:2605.03677, 2026.
13. "Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation." arXiv:2605.11739, 2026.
14. "Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation (G-OPD / ExOPD)." arXiv:2602.12125, 2026.

---

*整理时间：2026年7月 | 涵盖 arXiv、知乎、青稞社区、GitHub 等来源中 2025-2026 年的核心工作*
