export type ReadingStatus = "在读" | "计划中" | "已读";

export type ReadingItem = {
  title: string;
  author: string;
  kind: "书" | "论文" | "技术报告" | "教程";
  topic: string;
  status: ReadingStatus;
  note: string;
};

export const readingList: ReadingItem[] = [
  {
    title: "Reinforcement Learning: An Introduction",
    author: "Richard S. Sutton, Andrew G. Barto",
    kind: "书",
    topic: "RL 基础",
    status: "在读",
    note: "作为价值函数、策略梯度和探索问题的统一底稿，持续回读。",
  },
  {
    title: "OpenAI Spinning Up: PPO",
    author: "OpenAI",
    kind: "教程",
    topic: "PPO",
    status: "在读",
    note: "用来对照实现细节，尤其关注 advantage、clip 与 KL 约束。",
  },
  {
    title: "Direct Preference Optimization",
    author: "Rafailov et al.",
    kind: "论文",
    topic: "偏好优化",
    status: "已读",
    note: "适合和 RLHF 主线一起看，理解“无显式 reward model”的代价与收益。",
  },
  {
    title: "DeepSeek-R1 Technical Report",
    author: "DeepSeek-AI",
    kind: "技术报告",
    topic: "Reasoning",
    status: "已读",
    note: "重点记录 reasoning 数据构造、蒸馏路径和训练阶段拆分。",
  },
  {
    title: "GRPO / RLVR 相关实现与复现资料",
    author: "论文与开源实现",
    kind: "论文",
    topic: "后训练",
    status: "计划中",
    note: "准备专门整理一篇，比较它和 PPO 在稳定性、采样开销上的差异。",
  },
  {
    title: "Reward Modeling Survey",
    author: "综述论文",
    kind: "论文",
    topic: "Reward Model",
    status: "计划中",
    note: "希望从数据质量、标注协议和泛化误差三个维度做一份笔记。",
  },
];
