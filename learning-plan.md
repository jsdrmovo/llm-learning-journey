# 理论优先的 12 周 LLM 学习计划（含 Week 0 启动）

> 适配背景：偏理论推导；无本地算力；C 语言背景，能阅读 Python；目标是理解大模型工作全过程并用最小实验验证。

## Week 0：数学与反向传播快充（2–4 小时）
- 线代/概率/微积分核心公式速记；自动微分与计算图复习
- 产出：A4 速查表 + “待攻克清单”

## 第 1 周：Transformer 总览与注意力直觉
- 推导：缩放点积注意力、多头机制、残差与 LayerNorm
- 小实验：CPU 上跑单层注意力的字符级模型
- 产出：注意力为何有效（文字+图）

## 第 2 周：位置编码（绝对/相对/RoPE/ALiBi）
- 推导：RoPE 的旋转几何
- 小实验：替换位置编码，观察 PPL 变化
- 产出：RoPE 推导笔记

## 第 3 周：语言建模目标与训练稳定性
- 推导：交叉熵=最大似然；Teacher Forcing 本质
- 实践：极小 Transformer 在 Tiny 数据上收敛
- 产出：Loss 曲线与超参敏感性

## 第 4 周：分词与数据质量
- 原理：BPE/Unigram、字节级 BPE、中文混合场景
- 实践：自训小词表 + 数据统计
- 产出：分词器与数据管线脚本

## 第 5 周：优化与学习率
- 原理：AdamW/Lion、Warmup+Cosine、正则化
- 实验：优化器与调度对比
- 产出：对比报告

## 第 6 周：最小可训练 Transformer（里程碑 I）
- 实作：config 驱动训练循环 + 检查点 + PPL 评估
- 产出：可复现实验脚本与 README

## 第 7 周：推理与高效注意力概念
- 原理：KV Cache、FlashAttention（理解为主）
- 实验：开启/关闭 KV Cache 的延迟对比
- 产出：加速小结

## 第 8 周：参数高效微调（LoRA/QLoRA）
- 原理：低秩近似、量化直觉（NF4）
- 实验：对小模型做 LoRA SFT（小步数示范）
- 产出：显存/效果对比

## 第 9 周：量化与部署视图
- 原理：INT8/INT4（GPTQ/AWQ）、SmoothQuant；vLLM 认知
- 实验：8/4bit 量化并测延迟/吞吐
- 产出：部署能力矩阵

## 第 10 周：对齐与偏好优化（DPO/ORPO）
- 原理：SFT→RM→PPO 流程；无强化方法的权衡
- 实验：小规模 DPO 行为对比
- 产出：对齐前后案例对比

## 第 11 周：评测方法论
- 原理：MMLU/ARC/HellaSwag、MT-Bench、统计显著性
- 实践：轻量评测脚本
- 产出：评测清单与结果

## 第 12 周：总结与展望（里程碑 II）
- 产出：系统技术报告 + 下一阶段规划

## 建议资料
- 代码：karpathy/minGPT、micrograd、minbpe、llm.c
- 论文：Attention、Scaling Laws、Chinchilla、FlashAttention、LoRA/QLoRA、DPO/ORPO、量化（GPTQ/AWQ/SmoothQuant）
- 工具：PyTorch、transformers、datasets、accelerate、tokenizers、bitsandbytes、wandb

## 学习节奏
- 每周 6–10 小时：理论 2–3 + 实验 3–5 + 复盘 1
- 原则：先跑通极小版本，建立直觉，再做对比与推导收敛
