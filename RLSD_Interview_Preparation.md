# RLSD 面试准备手册

> 本文档为 LLM 算法实习面试准备，覆盖项目背景、核心算法、代码实现、面试深挖问题及参考答案。

---

## 目录

1. [项目概览与 Elevator Pitch](#1-项目概览与-elevator-pitch)
2. [背景知识：RLVR 与 GRPO](#2-背景知识rlvr-与-grpo)
3. [RLSD 核心算法详解](#3-rlsd-核心算法详解)
4. [理论贡献：OPSD 失败诊断](#4-理论贡献opsd-失败诊断)
5. [统一视角：GRPO / OPSD / RLSD](#5-统一视角grpo--opsd--rlsd)
6. [代码架构与关键实现](#6-代码架构与关键实现)
7. [实验结果与 Ablation 分析](#7-实验结果与-ablation-分析)
8. [面试深挖问题与参考答案](#8-面试深挖问题与参考答案)
9. [LLM 后训练通用面试题](#9-llm-后训练通用面试题)
10. [Agent 应用相关面试题](#10-agent-应用相关面试题)
11. [如何介绍这个项目](#11-如何介绍这个项目)

---

## 1. 项目概览与 Elevator Pitch

### 一句话总结

RLSD (RLVR with Self-Distillation) 将自蒸馏从分布匹配目标重新定义为 **token 级信用分配信号**，用环境奖励控制更新方向、用教师模型控制更新幅度，在 Qwen3-VL-8B 上平均提升 +4.69%。

### Elevator Pitch（30秒版本）

> 标准 GRPO 对所有 token 给相同的 sequence-level advantage，这很粗糙。RLSD 的核心洞察是：更新方向（该强化还是惩罚）必须来自可靠的环境奖励，但更新幅度（每个 token 该分配多少信用）可以从教师模型获得更细粒度的信号。教师模型接收特权信息（如 ground-truth answer），计算每个 token 的信息增益 Delta_t，然后根据环境奖励的正负方向来决定是"强化教师支持的 token"还是"惩罚教师反对的 token"。这是一个 drop-in replacement，不需要额外的辅助损失。

### 核心创新点

| 维度 | GRPO | OPSD (失败的自蒸馏) | RLSD (本文) |
|------|------|---------------------|-------------|
| 更新方向 | 环境奖励 | 教师模型 | 环境奖励 |
| 更新幅度 | 均匀 | 教师模型 | 教师模型 |
| 是否泄漏特权信息 | 否 | 是 | 否 |
| 需要的特权信息 | 无 | 完整推理轨迹 | 仅最终答案 |

---

## 2. 背景知识：RLVR 与 GRPO

### 2.1 RLVR (Reinforcement Learning with Verifiable Rewards)

RLVR 是一类后训练方法，核心思想是：
- 对于有可验证答案的任务（数学、代码、逻辑推理），可以用 **verifier** 自动判断模型输出的正确性
- 用这个 binary reward 作为 RL 信号来微调 LLM
- 不需要人类标注偏好数据，也不需要训练 reward model

**关键组件：**
- Policy model (pi_theta)：待优化的 LLM
- Verifier R(x, y)：判断 response y 对于 prompt x 是否正确，返回 0/1
- Rollout：从 policy 采样多个 response
- Advantage estimation：基于 reward 计算每个 response 的优势
- Policy gradient update：用 advantage 指导参数更新

### 2.2 GRPO (Group Relative Policy Optimization)

GRPO 是 DeepSeek 提出的 RL 算法，是 PPO 的简化版本：

**核心思想：**
- 对同一个 prompt x，采样 G 个 response {y^(1), ..., y^(G)}
- 用 verifier 对每个 response 打分得到 {R^(1), ..., R^(G)}
- 组内标准化得到 advantage：

```
A^(i) = (R^(i) - mean(R)) / (std(R) + eps)
```

- 策略梯度损失（带 PPO clipping）：

```
L_GRPO = E[ (1/G) * sum_i (1/|y^(i)|) * sum_t min(rho_t * A^(i), clip(rho_t, 1-eps, 1+eps) * A^(i)) ]
```

其中 `rho_t = pi_theta(y_t | x, y_{<t}) / pi_{theta_old}(y_t | x, y_{<t})` 是重要性比率。

**GRPO 的特点：**
- 不需要 Critic network（省显存、省计算）
- 组内标准化天然提供 baseline
- 对所有 token 给相同的 advantage（sequence-level）
- 简单有效，是当前 RLVR 的主流选择

**GRPO 的局限：**
- 所有 token 得到相同的 advantage，无法区分"关键 token"和"无关 token"
- 缺乏细粒度的 token-level credit assignment
- 训练后期容易 entropy collapse

---

## 3. RLSD 核心算法详解

### 3.1 核心洞察：方向与幅度的不对称需求

**关键观察：** 在策略梯度中，每个 token 的更新由两部分决定：
- **方向**（正/负）：决定是强化还是惩罚这个 token
- **幅度**（大/小）：决定更新多少

**不对称需求：**
- **方向信号**：可以稀疏，但 **必须可靠**。错误的方向会导致策略退化。环境奖励是可靠的，因为它来自外部验证。
- **幅度信号**：越密集越好，用于细粒度的 token 级信用分配。教师模型可以提供这个信号。

### 3.2 算法三步走

#### Step 1: 特权信息增益 (Privileged Information Gain)

```
Delta_t = sg( log P_T(y_t) - log P_S(y_t) )
```

- `P_S(y_t) = pi_theta(y_t | x, y_{<t})`：学生模型的预测
- `P_T(y_t) = pi_theta(y_t | x, r, y_{<t})`：教师模型的预测（r 是特权信息，如 ground-truth answer）
- `sg` = stop-gradient：梯度不流过 Delta_t

**物理含义：** Delta_t 度量了特权信息 r 对于预测 token y_t 的边际贡献。
- Delta_t > 0：教师（有答案）比学生（没答案）更确信这个 token → 这个 token 可能是关键的
- Delta_t < 0：教师反而不支持这个 token → 这个 token 可能是误导性的
- Delta_t ≈ 0：特权信息对这个 token 没有影响

#### Step 2: 方向感知的证据权重 (Direction-Aware Evidence Reweighting)

```
w_t = exp( sign(A) * Delta_t )
```

**关键设计：** 根据环境奖励 A 的正负来决定如何使用 Delta_t：
- **A > 0（正确 response）：** `w_t = exp(Delta_t) = P_T(y_t) / P_S(y_t)`
  - 教师支持的 token（Delta_t > 0）→ w_t > 1 → 放大 credit
  - 教师反对的 token（Delta_t < 0）→ w_t < 1 → 缩小 credit
- **A < 0（错误 response）：** `w_t = exp(-Delta_t) = P_S(y_t) / P_T(y_t)`
  - 教师反对的 token（Delta_t > 0）→ w_t < 1 → 缩小 blame
  - 教师支持的 token（Delta_t < 0）→ w_t > 1 → 放大 blame

**贝叶斯解释：** w_t 实际上是一个贝叶斯信念更新比率：

```
w_t = P(r | x, y_{<=t}) / P(r | x, y_{<t})
```

即"生成 token y_t 后，对特权信息 r 的信念更新了多少"。

#### Step 3: 裁剪的信用分配 (Clipped Credit Assignment)

```
A_hat_t = A * clip(w_t, 1-eps_w, 1+eps_w)
```

- 类似 PPO 的 ratio clipping，限制单个 token 的影响力
- 防止极端 token（如 Delta_t 很大）主导整个梯度
- eps_w 通常取 0.2，即 reweight 范围在 [0.8, 1.2]

#### 最终损失

```
L_RLSD = E[ (1/G) * sum_i (1/|y^(i)|) * sum_t min(rho_t * A_hat_t^(i), clip(rho_t, 1-eps, 1+eps) * A_hat_t^(i)) ]
```

与 GRPO 的唯一区别是 A 替换为 A_hat_t（token-level reweighted advantage）。

### 3.3 实现中的凸组合形式

代码中的实际实现使用了凸组合：

```
reweight_t = (1 - lambda) + lambda * clip(w_t, 1-eps_w, 1+eps_w)
A_hat_t = A * stop_gradient(reweight_t)
```

- lambda ∈ [0, 1]：控制 RLSD 信号的强度
- lambda = 0：退化为标准 GRPO
- lambda = 1：完全使用教师信号
- 通常使用 lambda warmup + decay schedule

### 3.4 自适应退课 (Implicit Curriculum)

**关键性质：** 随着训练进行，学生模型越来越接近教师模型：
- Delta_t = log P_T - log P_S → 0
- w_t → 1
- A_hat_t → A（标准 GRPO）

这意味着 RLSD **自动退化为 GRPO**，无需手动调参。早期训练（差距大）时教师信号强，后期训练（差距小）时教师信号弱。

### 3.5 Teacher Sync 策略

- 教师模型定期从 actor 同步权重（如每 10-20 步）
- 同步后教师 = 当前 actor，Delta_t 基于当前策略计算
- 不同步时教师是 frozen 的，提供稳定的 distillation 目标
- 比完全 frozen 的教师更好（避免 capacity ceiling）
- 比每步同步的教师更好（避免 instability）

---

## 4. 理论贡献：OPSD 失败诊断

### 4.1 OPSD (On-Policy Self-Distillation) 是什么

OPSD 让同一个模型同时扮演学生和教师：
- 学生：`P_S(. | x)` — 只看 prompt
- 教师：`P_T(. | x, r)` — 看 prompt + 特权信息 r
- 目标：最小化 `D_KL(P_T || P_S)` — 让学生模仿教师

### 4.2 为什么 OPSD 失败：KL 分解定理

**Theorem 1 (KL Decomposition):**

```
L_OPSD = L* + I(Y_t; R | X, Y_{<t})
```

其中：
- `L* = D_KL(P_bar_T || P_S)` — 理想目标（边际教师分布与学生的 KL）
- `I(Y_t; R | X, Y_{<t})` — 条件互信息，度量特权信息 r 对当前 token 预测的影响

**关键：** 第二项 `I(Y_t; R | X, Y_{<t})` **独立于 theta**，学生无法通过优化消除它。这是一个不可约的下界。

### 4.3 梯度分解

**Proposition (Per-Sample Gradient Decomposition):**

```
g(theta; r) = g*(theta) + delta(theta; r)
```

- `g*(theta)` = 边际匹配梯度（良性）
- `delta(theta; r)` = r 特定的偏差（病态）

**性质：**
- `E_r[delta] = 0`（零均值，不会系统性偏移）
- `E_r[||delta||^2] > 0`（正方差，会累积）
- 方差与 `I(Y_t; R | X)` 成正比 — 特权信息越有用，偏差越大

### 4.4 两阶段动态

**Phase 1（早期训练）：** 有益梯度主导，快速提升
- 学生从教师那里学到有用的推理模式
- Reward 快速上升

**Phase 2（后期训练）：** 偏差累积主导，性能退化
- 偏差驱动参数向利用特权信息的方向移动
- 模型开始"泄漏"特权信息（在没有答案时也能猜到答案）
- Reward 开始下降

### 4.5 不可能三角 (Impossibility Trilemma)

**Theorem：** 在共享参数的自蒸馏框架中，以下三个性质不可能同时满足：

| 策略 | 目标稳定性 | 持续改进 | 无泄漏 |
|------|-----------|----------|--------|
| Frozen Teacher | ✅ | ❌ (capacity ceiling) | ❌ (deviation accumulates) |
| Online Teacher | ❌ (teacher drift) | ✅ | ❌ (self-reinforcing feedback) |
| Periodic Sync | ⚠️ (intermittent) | ⚠️ | ❌ |
| **RLSD** | ✅ (anchored to R) | ✅ (lambda schedule) | ✅ (directional isolation) |

### 4.6 RLSD 为什么能解决

**三个隔离层级保证无泄漏：**

1. **方向隔离：** `sign(A_hat_t) = sign(A)` — 特权信息 r 无法改变梯度方向
2. **支持隔离：** 梯度只作用于学生采样的 token — 特权模式中的 token 采样概率为零
3. **幅度有界：** w_t 被裁剪到 `[1-eps, 1+eps]`，且被 lambda 衰减

---

## 5. 统一视角：GRPO / OPSD / RLSD

### 5.1 统一的策略梯度模板

三种方法都是以下模板的实例：

```
Delta theta ∝ E_{y~P_S}[ sum_t A_hat_t * nabla_theta log P_S(y_t | x, y_{<t}) ]
```

区别仅在于 A_hat_t 的定义：

| 方法 | A_hat_t | 方向来源 | 幅度来源 |
|------|---------|----------|----------|
| GRPO | A | 环境奖励 | 均匀 |
| OPSD (Rev. KL) | Delta_t = log P_T - log P_S | 教师 | 教师 |
| **RLSD** | `A * [(1-lam) + lam * clip(w_t)]` | **环境奖励** | **教师** |

### 5.2 设计哲学

- **GRPO：** "所有 token 同等重要" — 简单但粗糙
- **OPSD：** "教师知道所有答案" — 信息丰富但方向不可靠
- **RLSD：** "环境告诉我们对错，教师告诉我们为什么" — 最佳组合

---

## 6. 代码架构与关键实现

### 6.1 整体架构

```
RLSD/
├── verl/                          # 核心 RL 训练框架（基于 veRL 修改）
│   ├── trainer/
│   │   ├── core_algos.py          # 基础 RL 算法（GAE, GRPO, PPO loss）
│   │   ├── opsd_algos.py          # RLSD 专用算法（KL/JSD loss, teacher hint）
│   │   ├── opsd_config.py         # RLSD 配置参数
│   │   ├── opsd_trainer.py        # RLSD 训练循环
│   │   └── opsd_main.py           # 入口点
│   ├── workers/
│   │   ├── actor/
│   │   │   ├── dp_actor.py        # 标准 GRPO actor
│   │   │   └── dp_opsd_actor.py   # RLSD actor（核心 reweighting 逻辑）
│   │   ├── rollout/
│   │   │   └── vllm_rollout_spmd.py  # vLLM 推理引擎
│   │   └── fsdp_workers.py        # FSDP 分布式 worker
│   └── models/
│       └── transformers/
│           ├── qwen2_vl.py        # Qwen2-VL 适配
│           └── qwen3_vl.py        # Qwen3-VL 适配
├── examples/visual_rl/
│   ├── rlsd_config.yaml           # RLSD 训练配置
│   ├── rlsd_train.sh              # 训练启动脚本
│   ├── format_prompt/             # Jinja2 prompt 模板
│   └── reward_function/           # 奖励函数
└── paper/                         # 论文 LaTeX 源码
```

### 6.2 核心代码文件详解

#### `verl/trainer/core_algos.py` — 基础算法

```python
# GRPO Advantage 计算
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, eps=1e-6):
    # 1. 每个 sequence 的总 reward
    scores = token_level_rewards.sum(dim=-1)
    # 2. 组内标准化
    # A_i = (score_i - mean_group) / (std_group + eps)
    # 3. 广播到 token level
    return advantages, returns

# PPO Policy Loss（带 dual clip）
def compute_policy_loss(old_log_probs, log_probs, advantages, response_mask,
                        clip_ratio_low=0.2, clip_ratio_high=0.28, ...):
    # log_ratio = log_probs - old_log_probs
    # ratio = exp(clamp(log_ratio, -20, 20))
    # clipped_ratio = exp(clamp(log_ratio, log(1-eps_low), log(1+eps_high)))
    # DAPO-style dual clip for negative advantages
```

#### `verl/workers/actor/dp_opsd_actor.py` — RLSD 核心 reweighting

```python
@staticmethod
def _build_stgca_advantages(teacher_log_probs, student_log_probs, advantages,
                             response_mask, lam, clip_range, ...):
    # Step 1: 特权信息增益
    delta = teacher_log_probs - student_log_probs  # (B, T)

    # Step 2: 方向感知的证据权重
    sign_A = torch.sign(advantages)  # (B,)
    weight = torch.exp(sign_A.unsqueeze(-1) * delta)  # (B, T)

    # Step 3: 裁剪
    clipped_weight = torch.clamp(weight, 1 - clip_range, 1 + clip_range)

    # Step 4: 凸组合
    reweight = (1 - lam) + lam * clipped_weight  # (B, T)

    # Step 5: Stop-gradient + 应用
    reweighted_advantages = advantages * reweight.detach()

    return reweighted_advantages, metrics
```

#### `verl/trainer/opsd_trainer.py` — 训练循环

```python
# RaySimpleGRPOOPSDTrainer.fit() — 简化的 RLSD 流程
for step in range(total_steps):
    # 1. Student Rollout：采样 G 个 response
    batch = rollout(prompts)

    # 2. 计算标准 GRPO advantage
    advantages = compute_grpo_outcome_advantage(rewards)

    # 3. 构造 Teacher prompt（question + hint）
    teacher_prompts = construct_teacher_prompts(questions, hints, gt_answers)

    # 4. Teacher forward（no grad）计算 teacher log probs
    teacher_log_probs = teacher_forward(teacher_prompts, student_responses)

    # 5. RLSD reweighting
    effective_lambda = compute_lambda_schedule(step)
    reweighted_advantages = _build_stgca_advantages(
        teacher_log_probs, student_log_probs, advantages, lam=effective_lambda
    )

    # 6. Policy update
    loss = compute_policy_loss(old_log_probs, log_probs, reweighted_advantages)
    loss.backward()
    optimizer.step()

    # 7. 定期同步 teacher
    if step % teacher_sync_interval == 0:
        sync_teacher_from_actor()
```

### 6.3 关键实现细节

| 实现细节 | 位置 | 说明 |
|----------|------|------|
| Stop-gradient on Delta_t | `dp_opsd_actor.py:_build_stgca_advantages` | `reweight.detach()` 确保梯度不流过教师 |
| Dual clip (DAPO style) | `core_algos.py:compute_policy_loss` | 负 advantage 有额外的下界 clip |
| Lambda schedule | `opsd_trainer.py:_compute_lambda` | warmup → peak → decay → 0 |
| Teacher sync | `dp_opsd_actor.py:_sync_teacher_from_actor` | 定期从 actor 复制权重到 teacher |
| Thought-only reweighting | `dp_opsd_actor.py:_build_stgca_advantages` | 只在 `<thought>...</thought>` 内做 reweight |
| Negative-only reweighting | `dp_opsd_actor.py:_build_stgca_advantages` | 只对 A < 0 的 response 做 reweight |
| Top-k KL/JSD | `opsd_algos.py:compute_opsd_topk_kl_loss` | 只在 top-k logits 上计算 KL，节省显存 |
| Sequence length balancing | `utils/seqlen_balancing.py` | 按序列长度均衡分配到不同 GPU |

---

## 7. 实验结果与 Ablation 分析

### 7.1 主实验结果（Qwen3-VL-8B-Instruct）

| Method | MMMU | MathVista | MathVision | ZeroBench | WeMath | Avg |
|--------|------|-----------|------------|-----------|--------|-----|
| Base LLM | 62.44 | 73.80 | 47.37 | 19.76 | 54.10 | 51.49 |
| GRPO | 65.11 | 76.20 | 48.82 | 22.60 | 56.57 | 53.86 |
| OPSD | 63.82 | 75.10 | 47.53 | 21.06 | 54.95 | 52.49 |
| SDPO | 65.11 | 74.00 | 47.27 | 25.15 | 52.19 | 52.74 |
| GRPO+OPSD | 63.22 | 75.90 | 48.52 | 22.16 | 54.76 | 52.91 |
| **RLSD** | **67.22** | **78.10** | **52.73** | 24.85 | **58.00** | **56.18** |

**关键发现：**
- RLSD vs GRPO: +2.32% 平均提升
- RLSD vs OPSD: +3.69%（OPSD 训练会退化）
- RLSD vs GRPO+OPSD: +3.27%（加法融合有 scale mismatch 问题）
- 最大提升在 MathVision: +3.91%（竞赛级数学题，需要精细推理）

### 7.2 OPSD Pilot 实验

三种 OPSD 变体都显示了 **泄漏和退化**：
- **Full OPSD：** 最宽泄漏带宽，中等泄漏率
- **Teacher's Top-1：** 最窄带宽但最集中注入特权信息，**泄漏最严重**
- **Student's Top-1：** 最低但仍非零泄漏

**结论：** 任何让 `P_T(.|r)` 进入梯度方向的分布匹配方法都会泄漏。

### 7.3 训练动态分析

1. **Reward 曲线：** RLSD 更陡峭的初始上升，避免 OPSD 的后期崩溃
2. **Entropy：** GRPO 快速 entropy collapse；RLSD 保持更高 entropy（选择性强化关键 token 而非均匀压制）
3. **Clip ratio：** 稳定在 3%-6%，说明 clipping 机制在起作用

### 7.4 Case Study（Token-level Credit Heatmap）

- **正确 response：** RLSD 将更大 credit 集中在关键推理步骤（识别物体、执行计算），降低通用叙述的权重
- **错误 response：** 最大 blame 分配给具体的错误关系和错误答案，中性的 setup token 受到较小惩罚

---

## 8. 面试深挖问题与参考答案

### 8.1 项目理解类

**Q1: 用 3 分钟介绍你的 RLSD 项目**

> 参考 [第 11 节](#11-如何介绍这个项目)

**Q2: RLSD 和标准 GRPO 的核心区别是什么？**

> 核心区别在于 **token-level credit assignment**。GRPO 对所有 token 给相同的 sequence-level advantage（binary reward 的组内标准化），无法区分关键 token 和无关 token。RLSD 引入教师模型（接收特权信息如 ground-truth answer）来计算每个 token 的信息增益 Delta_t，然后用这个信号来 reweight advantage：对正确 response，教师支持的 token 得到更大 credit；对错误 response，教师反对的 token 得到更大 blame。方向始终由环境奖励控制，教师只控制幅度。

**Q3: 为什么不用 OPSD（直接做自蒸馏），而要用 RLSD？**

> OPSD 有一个根本性问题：它让教师的分布 P_T(.|r) 直接决定梯度方向。我们证明了这会导致一个不可约的互信息间隙 I(Y_t; R | X)，学生永远无法消除。更严重的是，梯度中的 r-specific 偏差会随训练累积，导致 **特权信息泄漏** — 模型在没有答案时也能猜到答案。RLSD 的关键设计是：环境奖励决定方向（正/负），教师只决定幅度（大/小），且幅度被裁剪到 [1-eps, 1+eps] 的范围内，结构上避免了泄漏。

**Q4: 为什么 RLSD 中的 w_t 要乘以 sign(A)？**

> 这是 direction-aware 的设计。对于正确 response（A > 0），我们想强化教师支持的 token，所以 w_t = P_T/P_S（教师概率高的 token 权重更大）。对于错误 response（A < 0），我们想惩罚导致错误的 token，所以 w_t = P_S/P_T（学生偏离教师的 token 权重更大，因为这些 token 更可能是错误的根源）。sign(A) 确保了 **环境奖励始终控制更新方向**，教师只影响幅度。

**Q5: 为什么 RLSD 要用 clip(w_t, 1-eps, 1+eps)？**

> 两个原因：
> 1. **稳定性：** 某些 token 的 Delta_t 可能非常大（如学生完全预测错误的 token），导致 w_t 极端。Clipping 限制了单个 token 的影响力，类似 PPO 的 ratio clipping。
> 2. **信任域约束：** 限制 reweight 范围确保更新不会太大，维持训练的稳定性。
> 
> 实际上 eps_w 通常等于 PPO 的 clip ratio（0.2），所以 reweight 范围是 [0.8, 1.2]。

**Q6: 为什么 RLSD 有 lambda schedule（warmup + decay）？**

> Lambda 控制 RLSD 信号的强度。Warmup 避免训练初期教师信号太强（此时教师和学生差距大，Delta_t 大）。Decay 的设计很有意思：随着训练进行，学生接近教师，Delta_t 自然衰减，RLSD 自动退化为 GRPO。手动 decay 是一个额外的安全措施，确保后期不会过度依赖教师信号。

**Q7: Teacher Sync 策略是怎么工作的？为什么不同步更新？**

> Teacher 每 N 步（如 10 或 20 步）从 actor 同步一次权重。不同步时 teacher 是 frozen 的。
> 
> 为什么不同步更新？因为完全同步的 teacher（= online teacher）会导致 **目标漂移**：每步优化后，teacher 也变了，学生追逐的是一个移动的目标。Frozen teacher 提供稳定的目标，但会有 capacity ceiling（不能超过初始 checkpoint 的能力）。Periodic sync 是折中方案：大部分时间提供稳定目标，偶尔更新以跟上学生的进步。

### 8.2 技术细节类

**Q8: Delta_t = sg(log P_T - log P_S) 中的 stop-gradient 为什么重要？**

> 如果不用 stop-gradient，梯度会通过 Delta_t 流向教师模型。但在 RLSD 中，教师和学生共享参数（或教师是从 actor 复制的），这会导致：
> 1. 梯度通过 Delta_t 优化的是"让 Delta_t 更大"，而不是"让学生更好"
> 2. 本质上变成了一个分布匹配目标，重新引入 OPSD 的问题
> 
> Stop-gradient 确保 Delta_t 只作为一个 **标量权重** 来 rescale advantage，梯度只通过 student log probs 流动。

**Q9: RLSD 中的贝叶斯解释是什么？**

> 在 Consistent Conditional Approximation 假设下：
> ```
> w_t = P_T(y_t | x, r, y_{<t}) / P_S(y_t | x, y_{<t})
>     = P(r | x, y_{<=t}) / P(r | x, y_{<t})
> ```
> 即 w_t 是一个 **贝叶斯信念更新比率**：生成 token y_t 后，对特权信息 r 的信念更新了多少。
> 
> 序列层面的 telescoping：
> ```
> prod_{t=1}^{T} w_t = P(r | x, y) / P(r | x)
> ```
> 即整个序列的权重乘积等于"看到完整 response 后，对 r 的后验/先验比率"。

**Q10: 为什么 RLSD 的 gradient 不存在 deviation 问题？**

> OPSD 的梯度分解为 `g = g* + delta`，其中 delta 是 r-specific 的偏差。RLSD 的梯度是：
> ```
> nabla L_RLSD = E_y[ sum_t A_hat_t * nabla log P_S(y_t) ]
> ```
> 其中 A_hat_t = A * reweight_t。这里的 A 来自环境奖励（对所有 r 是一致的），reweight_t 通过 stop-gradient 确保不引入额外梯度。因此梯度方向完全由 A 决定，不存在 r-specific 的偏差。

**Q11: RLSD 的隐式退课 (implicit curriculum) 是怎么实现的？**

> 随着训练进行：
> 1. 学生模型越来越接近教师模型
> 2. Delta_t = log P_T - log P_S → 0
> 3. w_t = exp(sign(A) * Delta_t) → 1
> 4. A_hat_t → A（标准 GRPO）
> 
> 这意味着 RLSD 在训练后期自动退化为 GRPO，教师信号的影响自然消退。这是一个 **不需要手动调参的自适应机制**。

**Q12: 代码中 RLSD reweighting 的凸组合形式 `(1-lambda) + lambda * clip(w_t)` 和论文中的 `A * clip(w_t)` 有什么区别？**

> 论文中简化了表述。代码中的凸组合形式是一个更灵活的实现：
> - lambda = 0：完全退化为 GRPO
> - lambda = 1：论文中的标准 RLSD
> - lambda ∈ (0, 1)：混合版本
> 
> 凸组合还有一个数值稳定性的优势：`(1-lambda) + lambda * clip(w_t)` 保证 reweight 在 `[1 - lambda*eps, 1 + lambda*eps]` 范围内，比直接用 `clip(w_t, 1-eps, 1+eps)` 更保守。

### 8.3 对比分析类

**Q13: RLSD 和 PPO 的 Critic 有什么区别？**

> PPO 的 Critic 估计 V(s) = E[sum future rewards | state]，用于计算 GAE advantage = reward - V(s)。这是一个 **learned baseline**，需要额外的 value network。
> 
> RLSD 的 teacher 信号是 **stop-gradient 的 log-probability ratio**，不是 learned value。它不需要额外网络（teacher 和 actor 共享参数），只额外需要一次 teacher forward pass。而且 RLSD 的信号是 token-level 的，而 PPO Critic 通常是 state-level 的。

**Q14: RLSD 和 Reward Shaping 有什么区别？**

> Reward shaping 修改环境奖励函数（如添加 intermediate rewards），而 RLSD 不修改奖励。RLSD 的 advantage 仍然是标准的 GRPO advantage（来自环境奖励），只是被 teacher 信号 reweight。
> 
> 关键区别：Reward shaping 改变了优化目标，可能导致 reward hacking。RLSD 不改变优化目标（仍然是最大化环境奖励），只是改变了 **信用分配**（哪些 token 更重要）。

**Q15: RLSD 和 DPO/KTO 等 offline 方法相比有什么优势？**

> 1. **Online vs Offline：** RLSD 是 on-policy 的，每步用当前策略采样；DPO 是 off-policy 的，依赖静态数据集
> 2. **Token-level vs Sequence-level：** RLSD 提供 token-level 的信用分配；DPO 只有序列级别的 preference signal
> 3. **不需要 preference pairs：** RLSD 只需要 verifier（0/1 reward），不需要人类标注的 (chosen, rejected) pairs
> 4. **更好的探索：** on-policy 采样允许模型探索新的解题路径

**Q16: GRPO+OPSD（加法融合）为什么不如 RLSD？**

> GRPO+OPSD 的损失是：
> ```
> L = L_GRPO + alpha * L_OPSD
> ```
> 问题在于：
> 1. **Scale mismatch：** L_GRPO 是 policy gradient（有正负，量级小），L_OPSD 是 KL divergence（非负，量级大）。两者相加会导致 KL loss 主导梯度。
> 2. **方向冲突：** L_GRPO 的梯度方向由环境奖励决定，L_OPSD 的梯度方向由教师决定。两者可能冲突。
> 3. **OPSD 的固有问题：** L_OPSD 本身就有泄漏问题，加法融合无法解决。
> 
> RLSD 用 **乘法调制** 避免了这些问题：reweight 只改变 advantage 的幅度，不改变方向，不引入额外损失。

### 8.4 实验与 Ablation 类

**Q17: 为什么选择 Qwen3-VL-8B 作为 base model？**

> 1. 多模态推理是当前的热点方向
> 2. 8B 规模适合学术资源（4 节点 x 8 H200）
> 3. Qwen3-VL 是当时最先进的开源多模态模型之一
> 4. 便于与已有工作（如 EasyVideoR1）对比

**Q18: 为什么选择 MMMU、MathVista 等 benchmark？**

> 这些 benchmark 覆盖了多模态推理的不同维度：
> - **MMMU：** 多学科（科学、工程、人文）大学水平
> - **MathVista：** 视觉数学推理
> - **MathVision：** 竞赛级视觉数学
> - **ZeroBench：** 压力测试，设计为前沿模型无法解决
> - **WeMath：** 细粒度数学问题求解
> 
> 数学类 benchmark 特别适合评估 RLVR，因为答案可验证。

**Q19: RLSD 在 ZeroBench 上的提升不如其他 benchmark 显著，为什么？**

> ZeroBench 设计为"前沿模型无法解决"的题目。这暗示：
> 1. 这些题目可能超出了 8B 模型的能力上限
> 2. Teacher 信号（来自同一个 8B 模型）在这些难题上也不够可靠
> 3. RLSD 的优势在于 **精细的信用分配**，但如果题目本身太难（正确率极低），信用分配的收益有限
> 
> 这也暗示了 RLSD 的一个局限：教师的能力上限约束了学生的学习上限。

**Q20: Ablation 中为什么 Teacher's Top-1 比 Full OPSD 泄漏更严重？**

> Teacher's Top-1 将 distillation 目标压缩为 `argmax_v P_T(v|r)`，即只保留教师最确信的 token。
> 
> 乍看之下，这应该减少泄漏（只传递最强信号）。但实际上：
> 1. **更集中的注入：** Full OPSD 的 KL 覆盖整个 vocabulary，信号分散；Top-1 只有一个 token，信号高度集中
> 2. **更强的梯度：** 集中的 signal 意味着更大的 per-token gradient，更快地驱动参数向利用 r 的方向移动
> 3. **更少的噪声：** Full OPSD 的 gradient 有大量 vocabulary 上的噪声；Top-1 几乎无噪声，偏差累积更快
> 
> 这验证了理论预测：只要 P_T(.|r) 进入梯度方向，任何压缩策略都无法避免泄漏。

---

## 9. LLM 后训练通用面试题

### 9.1 RLHF / RLAIF 基础

**Q: RLHF 的基本流程是什么？**
> 1. SFT（Supervised Fine-Tuning）：用高质量数据微调 base model
> 2. Reward Model Training：用人类偏好数据训练 reward model
> 3. RL Optimization：用 PPO/GRPO 优化 policy 以最大化 reward model 的分数

**Q: PPO 在 LLM 训练中有哪些问题？**
> 1. 需要额外的 Critic network（显存开销大）
> 2. Reward hacking（policy 学会 exploit reward model 的弱点）
> 3. KL 散度约束的调参困难
> 4. 训练不稳定（reward hacking 导致的 mode collapse）

**Q: GRPO 相比 PPO 的优势是什么？**
> 1. 不需要 Critic（省显存、省计算）
> 2. 组内标准化天然提供 baseline
> 3. 实现更简单，训练更稳定
> 4. 适合可验证奖励场景（不需要 learned reward model）

**Q: DPO 的原理是什么？和 RLHF 有什么区别？**
> DPO 将 reward model training 和 policy optimization 合并为一步：
> ```
> L_DPO = -log(sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x))))
> ```
> 区别：DPO 不需要显式的 reward model，直接从 preference pairs 优化 policy。但 DPO 是 off-policy 的，依赖静态数据。

### 9.2 后训练技术

**Q: 什么是 SFT？为什么需要 SFT？**
> SFT (Supervised Fine-Tuning) 用高质量的 instruction-response 对微调 base model。
> 需要 SFT 的原因：base model 是 next-token predictor，不是 instruction follower。SFT 教会模型"对话格式"和"遵循指令"。

**Q: 什么是 LoRA？为什么有效？**
> LoRA (Low-Rank Adaptation) 冻结原始权重，只训练低秩增量：
> ```
> W' = W + A * B,  where A in R^{d x r}, B in R^{r x d}, r << d
> ```
> 有效的原因：
> 1. 参数效率高（r 通常 8-64）
> 2. 不改变原始权重，避免 catastrophic forgetting
> 3. 可以针对不同任务训练不同的 LoRA adapters

**Q: 什么是 reward hacking？如何缓解？**
> Reward hacking 是 policy 学会 exploit reward model 的弱点，获得高 reward 但实际质量下降。
> 缓解方法：
> 1. KL 约束（限制 policy 偏离 reference model 的程度）
> 2. Reward model ensemble
> 3. Iterative RLHF（定期更新 reward model）
> 4. 使用 verifiable rewards（如 RLSD）

**Q: 什么是 entropy collapse？为什么在 GRPO 中常见？**
> Entropy collapse 是 policy 的输出分布变得过于确定（低 entropy），丧失探索能力。
> 在 GRPO 中常见是因为：sequence-level reward 对所有 token 施加相同的正/负 pressure，导致所有 token 的概率都被推向极端。
> RLSD 的优势之一是通过 token-level reweighting 缓解了这个问题：只强化关键 token，不均匀压制所有 token。

### 9.3 训练技术

**Q: 什么是 FSDP？和 DeepSpeed ZeRO 有什么区别？**
> FSDP (Fully Sharded Data Parallelism) 是 PyTorch 原生的分布式训练策略，将模型参数、梯度、优化器状态分片到多个 GPU。
> 与 DeepSpeed ZeRO 的区别：
> 1. FSDP 是 PyTorch 原生，DeepSpeed 是第三方库
> 2. FSDP 使用 PyTorch 的 DTensor，DeepSpeed 使用自己的分片格式
> 3. 性能上差异不大，但 FSDP 与 PyTorch 生态集成更好

**Q: 什么是 vLLM？为什么用于 rollout？**
> vLLM 是高性能的 LLM 推理引擎，核心创新是 PagedAttention（类似操作系统虚拟内存的 KV cache 管理）。
> 用于 rollout 的原因：
> 1. 推理速度快（比 naive 实现快 2-4x）
> 2. 支持 continuous batching（动态 batching 提高 GPU 利用率）
> 3. 支持多种 sampling 策略

**Q: 什么是 gradient accumulation？为什么需要？**
> Gradient accumulation 是在多个 micro-batch 上累积梯度，然后再做一次 optimizer step。
> 需要的原因：当 effective batch size 很大（如 256）但 GPU 显存不够放整个 batch 时，需要将 batch 拆成多个 micro-batch。

**Q: 什么是 mixed precision training？有什么好处？**
> 使用 FP16/BF16 进行前向/反向传播，FP32 维护 master weights 和 optimizer states。
> 好处：
> 1. 显存减半（可以训更大模型或更大 batch）
> 2. 计算加速（Tensor Core 对 FP16/BF16 优化）
> 3. 通信量减半（分布式训练中梯度通信）

---

## 10. Agent 应用相关面试题

### 10.1 Agent 基础

**Q: 什么是 LLM Agent？和普通 LLM 应用有什么区别？**
> LLM Agent 是一个具有 **自主决策能力** 的系统，LLM 作为核心"大脑"来：
> 1. 理解任务和环境
> 2. 规划执行步骤
> 3. 调用工具（搜索、代码执行、API 调用）
> 4. 观察结果并调整策略
> 
> 普通 LLM 应用是单轮/多轮对话，Agent 是一个 **循环的 观察-思考-行动 循环**。

**Q: Agent 的核心组件有哪些？**
> 1. **Planning：** 任务分解、步骤规划（CoT, ReAct, Tree of Thoughts）
> 2. **Memory：** 短期（对话历史）和长期（向量数据库）记忆
> 3. **Tool Use：** 调用外部工具扩展能力
> 4. **Reflection：** 评估执行结果，调整策略

**Q: ReAct 是什么？**
> ReAct (Reasoning + Acting) 是一个 prompting 框架：
> ```
> Thought: 我需要搜索这个信息
> Action: search("query")
> Observation: [search results]
> Thought: 根据搜索结果，答案是...
> Action: finish("answer")
> ```
> 核心是 interleave reasoning 和 acting，让模型在每一步都能"思考"和"行动"。

### 10.2 Agent 与后训练

**Q: RLSD 的思想可以应用到 Agent 训练吗？**
> 可以。Agent 训练中的一个核心问题是 **多步决策的信用分配**：一个成功的最终结果应该归功于哪些中间步骤？
> 
> RLSD 的思想可以推广：
> - **环境奖励** = 最终任务是否成功
> - **教师信号** = 每个中间步骤的"质量"评估（可以来自更强的模型、人类标注、或自动评估）
> - **RLSD reweighting** = 对关键步骤给更大 credit，对无关步骤给更小 credit
> 
> 这比 uniform advantage（所有步骤相同 credit）更精细。

**Q: Agent 训练中的 reward 设计有哪些挑战？**
> 1. **稀疏奖励：** 只有最终结果有 reward，中间步骤没有
> 2. **延迟奖励：** 某个步骤的影响可能在很多步之后才显现
> 3. **奖励欺骗：** Agent 可能找到捷径完成任务但跳过了关键步骤
> 4. **多目标冲突：** 任务完成度、效率、安全性可能冲突

**Q: 如何用 RL 训练 Agent 的 tool use 能力？**
> 1. **环境构建：** 定义工具接口和可验证的任务
> 2. **奖励设计：** binary（任务完成）+ dense（中间步骤质量）
> 3. **探索策略：** epsilon-greedy、curiosity-driven exploration
> 4. **信用分配：** 如 RLSD 的 token-level reweighting，或 Monte Carlo rollout
> 5. **课程学习：** 从简单工具使用到复杂工具组合

### 10.3 多模态 Agent

**Q: 多模态 Agent 和纯文本 Agent 有什么区别？**
> 1. **感知能力：** 需要理解图像、视频等视觉输入
> 2. **Grounding：** 将语言指令与视觉实体对应
> 3. **Action space：** 可能包括点击、拖拽等 GUI 操作
> 4. **更长的 context：** 视觉 token 通常很长

**Q: RLSD 为什么选择多模态场景（Qwen3-VL）？**
> 1. 多模态推理是当前热点
> 2. 视觉理解需要精细的 token-level credit assignment（识别关键视觉区域）
> 3. 多模态任务的 verifier 更丰富（数学公式、OCR、物体检测都有标准评估）

---

## 11. 如何介绍这个项目

### 11.1 一分钟版本

> 我参与的 RLSD 项目解决的是 LLM 后训练中的 **token-level credit assignment** 问题。在标准的 GRPO 算法中，所有 token 得到相同的 advantage，这很粗糙。RLSD 的核心洞察是：我们可以用一个教师模型（接收特权信息如 ground-truth answer）来计算每个 token 的信息增益，然后用这个信号来 reweight advantage。关键设计是 **方向由环境奖励控制，幅度由教师控制**，并通过 stop-gradient 和 clipping 保证训练稳定。
> 
> 理论上，我们证明了之前的自蒸馏方法（OPSD）会因为梯度中的 r-specific 偏差而失败，导致特权信息泄漏。RLSD 通过将教师从"生成目标"重新定位为"幅度评估器"来解决这个问题。
> 
> 实验上，RLSD 在 Qwen3-VL-8B 上平均提升 4.69%，特别是在数学推理任务上提升显著。

### 11.2 三分钟版本

> **问题定义：**
> LLM 后训练中，RLVR（Reinforcement Learning with Verifiable Rewards）是一个重要方向，用环境奖励（如数学答案的正确性）来优化模型。主流算法 GRPO 对所有 token 给相同的 advantage，无法区分关键 token 和无关 token。
> 
> **核心方法：**
> RLSD 引入一个教师模型（与学生共享参数，但接收特权信息如 ground-truth answer）来实现 token-level credit assignment。具体来说：
> 1. 计算每个 token 的特权信息增益 Delta_t = sg(log P_T - log P_S)
> 2. 根据环境奖励的正负方向，构造方向感知的权重 w_t = exp(sign(A) * Delta_t)
> 3. 裁剪后 reweight advantage：A_hat_t = A * clip(w_t, 1-eps, 1+eps)
> 
> **理论贡献：**
> 我们证明了之前的自蒸馏方法 OPSD 会失败，因为其梯度包含一个不可约的 r-specific 偏差，会导致特权信息泄漏。我们提出了一个不可能三角：目标稳定性、持续改进、无泄漏不可能同时满足。RLSD 通过"方向来自环境、幅度来自教师"的设计，同时满足三个性质。
> 
> **实验结果：**
> 在 Qwen3-VL-8B-Instruct 上，RLSD 平均提升 4.69%（vs Base），2.32%（vs GRPO），特别在 MathVision 上提升 3.91%。Ablation 证明了 RLSD 的隐式退课、entropy 维持、和泄漏免疫等性质。

### 11.3 准备好的追问

**"你的贡献是什么？"**
> 1. 实现了核心的 RLSD reweighting 逻辑（`dp_opsd_actor.py:_build_stgca_advantages`）
> 2. 设计了 Teacher Sync 策略和 lambda schedule
> 3. 实现了 thought-only reweighting（只在 `<thought>` 标签内做 reweight）
> 4. 参与了多模态 reward function 的设计和调试
> 5. 运行了主要实验和 ablation

**"遇到了什么困难？"**
> 1. **Scale mismatch：** 最初尝试 GRPO+OPSD 加法融合，发现 KL loss 和 policy gradient 的 scale 差异很大，导致训练不稳定。后来改用乘法调制（RLSD）解决了这个问题。
> 2. **Teacher sync 频率：** 每步同步导致目标漂移，不同步导致 capacity ceiling。最终通过实验确定了 10-20 步的 sync 间隔。
> 3. **显存优化：** Teacher forward 需要额外的显存。通过 top-k KL 近似和 gradient checkpointing 解决。

**"如果继续做，你会怎么改进？"**
> 1. **更强的 teacher：** 当前 teacher 和 student 是同一个模型。可以用更强的外部模型作为 teacher（如 GPT-4），但需要解决 off-policy 问题。
> 2. **Adaptive lambda：** 当前的 lambda schedule 是预设的。可以根据 Delta_t 的统计量自适应调整。
> 3. **扩展到 Agent：** 将 RLSD 的 token-level credit assignment 应用到多步决策的 Agent 训练中。

---

## 附录：关键术语表

| 术语 | 含义 |
|------|------|
| RLVR | Reinforcement Learning with Verifiable Rewards |
| GRPO | Group Relative Policy Optimization |
| OPSD | On-Policy Self-Distillation |
| RLSD | RLVR with Self-Distillation |
| SDPO | Self-Distilled Policy Optimization |
| PPO | Proximal Policy Optimization |
| KL | Kullback-Leibler Divergence |
| JSD | Jensen-Shannon Divergence |
| FSDP | Fully Sharded Data Parallelism |
| vLLM | High-performance LLM inference engine |
| Advantage | A = (R - baseline) / std，衡量一个 response 的相对优势 |
| Credit Assignment | 将最终 reward 分配到每个 token 的过程 |
| Privileged Information | 教师有但学生没有的信息（如 ground-truth answer） |
| Information Gain | Delta_t = log P_T - log P_S，特权信息对 token 预测的影响 |
| Trust Region | 限制每步更新幅度的机制（如 PPO clipping） |
| Entropy Collapse | Policy 输出分布过于确定，丧失探索能力 |
| Reward Hacking | Policy exploit reward model 的弱点 |
| Curriculum Learning | 从简单到复杂的课程式训练 |

---

## 附录：推荐阅读

1. **DeepSeek-R1:** RLVR 的开创性工作
2. **GRPO 论文:** Group Relative Policy Optimization
3. **PPO 论文:** Proximal Policy Optimization Algorithms
4. **DPO 论文:** Direct Preference Optimization
5. **veRL 框架:** RLSD 的基础框架
6. **EasyR1/EasyVideoR1:** RLSD 的上游项目
