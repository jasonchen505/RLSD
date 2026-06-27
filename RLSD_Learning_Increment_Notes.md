# RLSD 学习增量笔记：从复现计划中学到的新知识点

> 本文档记录在制定 8x 3090 复现计划过程中，对比前两轮（项目理解 + 面试准备）增量新学到的点。

---

## 1. 工程约束与理论设计的冲突

### 1.1 FSDP CPU Offload 与 Gradient Accumulation 的互斥

**代码位置：** `verl/workers/fsdp_workers.py:148-152`

```python
if (
    config.fsdp.enable_cpu_offload
    and config.global_batch_size_per_device != config.micro_batch_size_per_device_for_update
):
    raise ValueError(f"{role} cannot use FSDP's CPU offload when gradient accumulation is enabled.")
```

**新学到的点：**

FSDP 的 CPU offload 要求每个 micro-batch 完整地在 GPU 上完成 forward + backward，然后才能 offload 回 CPU。但 gradient accumulation 需要多个 micro-batch 累积梯度后再做 optimizer step。这两个机制冲突：

- CPU offload 需要：load → forward → backward → offload → load → forward → backward → offload → optimizer step
- Gradient accumulation 需要：load → forward → backward → load → forward → backward → optimizer step → offload

前者的 load/offload 开销太大，后者无法在多个 micro-batch 间保持模型在 GPU 上。

**实际影响：** 在 3090 上，必须用小 batch size（等于 micro batch size），不能用 gradient accumulation 来模拟大 batch。

**面试价值：** 这是一个很好的"理论可行但工程不可行"的例子。理论上 gradient accumulation 可以模拟任意大 batch，但工程上与 CPU offload 冲突。

### 1.2 vLLM 与 FSDP 的显存竞争

**新学到的点：**

训练流程中，vLLM rollout 引擎和 FSDP 训练模型需要 **分时复用** GPU 显存：

```
Rollout 阶段: vLLM 加载到 GPU → 生成 response → vLLM offload
Training 阶段: FSDP 模型加载到 GPU → forward/backward → FSDP offload
```

代码中通过 `prepare_rollout_engine()` 和 `release_rollout_engine()` 管理这个生命周期。

**实际影响：** 在 24GB 的 3090 上，需要精确控制 `gpu_memory_utilization` 给 vLLM 分配的显存，留够 FSDP 训练用的空间。

**面试价值：** 展示了对 hybrid engine 架构的理解——同一个 GPU 需要在推理引擎和训练框架之间切换。

### 1.3 3090 的 PCIe 通信瓶颈

**新学到的点：**

原始配置用 4 节点 x 8 H200，有 NVLink（900GB/s）互联。3090 只有 PCIe 4.0（64GB/s），差 14 倍。

FSDP 的每个 forward/backward 都需要 all-gather 参数，每个 optimizer step 都需要 reduce-scatter 梯度。这些 collective 操作在 PCIe 上会成为瓶颈。

**实际影响：**
1. 训练速度比 H200 慢 2-3x（不仅是计算慢，通信也慢）
2. `tensor_parallel_size` 必须设为 1（跨 GPU 的 tensor parallel 在 PCIe 上太慢）
3. FSDP full shard 的通信开销比预期大

**面试价值：** 理解分布式训练中的通信瓶颈，不只是计算瓶颈。

---

## 2. 显存工程的精细计算

### 2.1 8B 模型在 FSDP 下的显存分解

**新学到的点：**

一个 8B 参数的模型（BF16），FSDP full shard 到 8 GPU 上：

| 组件 | 全量大小 | 每卡分片（8 GPU） |
|------|---------|------------------|
| 模型参数 (BF16) | 16 GB | 2 GB |
| AdamW momentum (FP32) | 16 GB | 2 GB |
| AdamW variance (FP32) | 16 GB | 2 GB |
| 梯度 (BF16) | 16 GB | 2 GB |
| **总计** | **64 GB** | **8 GB** |

加上激活值（gradient checkpointing 后约 3-5 GB）和 overhead，每卡需要 ~12-15 GB。

**实际影响：** 8B 模型在 8x 3090 上训练是 **理论可行** 的，但需要 CPU offload + gradient checkpointing + 小 batch。

**面试价值：** 展示了对 FSDP 显存分解的精确理解，不只是"模型多大显存多大"的粗略估计。

### 2.2 vLLM 的 KV Cache 显存计算

**新学到的点：**

vLLM 的 `gpu_memory_utilization` 控制的是 **总显存的百分比**，不是增量。在 24GB 的 3090 上：

- `gpu_memory_utilization=0.4` → vLLM 使用 ~9.6 GB
- `gpu_memory_utilization=0.6` → vLLM 使用 ~14.4 GB

但 vLLM 加载时，FSDP 模型已经被 offload 了，所以 vLLM 可以用大部分显存。关键是 rollout 结束后，FSDP 模型加载回来时，vLLM 必须已经释放显存。

**实际影响：** 需要确保 `release_rollout_engine()` 真正释放了 vLLM 的显存，否则 FSDP 加载时会 OOM。

### 2.3 `expandable_segments:True` 的作用

**新学到的点：**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

PyTorch 的 CUDA 内存分配器默认使用固定大小的 segment。当频繁分配/释放不同大小的 tensor 时，会产生大量碎片，导致"有足够总显存但没有连续空间"的 OOM。

`expandable_segments:True` 让分配器使用可扩展的 segment，减少碎片。

**实际影响：** 在 24GB 的 3090 上，显存碎片问题更严重（总空间小，碎片更容易导致 OOM）。这个环境变量是必须的。

**面试价值：** 展示了对 CUDA 内存管理的理解，不只是 PyTorch 层面。

---

## 3. 分布式同步的工程细节

### 3.1 `all_reduce(MIN)` 确保 Batch Size 一致

**代码位置：** `verl/workers/actor/dp_opsd_actor.py:730-736`

```python
on_batch_size_t = torch.tensor(on_batch_size, device=device, dtype=torch.long)
dist.all_reduce(on_batch_size_t, op=dist.ReduceOp.MIN)
on_batch_size = int(on_batch_size_t.item())
```

**新学到的点：**

在 RLSD 中，不同 GPU 可能有不同数量的有效 on-policy/off-policy 样本（因为 teacher 过滤掉了 reward=0 的样本）。但 NCCL 的 collective 操作要求所有 rank 执行相同次数。

解决方案：取所有 rank 的 batch size 的 **最小值**。如果某个 rank 有 0 个有效样本，所有 rank 都变成 0。

**实际影响：** 在小 batch size（如 32）的情况下，某些 step 可能所有样本都被过滤掉，导致该 step 完全没有梯度更新。

**面试价值：** 这是一个很好的"分布式训练中的 edge case"案例。

### 3.2 Dummy Forward-Backward 保持 Collective 同步

**代码位置：** `verl/workers/actor/dp_opsd_actor.py:833-861`

```python
(dummy_student_logits.sum() * 0.0).backward()
```

**新学到的点：**

当某个 rank 没有有效数据时，不能直接 skip（会导致 NCCL hang），也不能不做 backward（会导致 FSDP 的 all-gather/reduce-scatter 不匹配）。

解决方案：执行一个 dummy forward + backward，loss 乘以 0.0，保证：
1. 计算图被创建（FSDP 的 all-gather 触发）
2. backward 被执行（FSDP 的 reduce-scatter 触发）
3. 梯度为零（不影响模型更新）

**面试价值：** 展示了对 FSDP 内部机制的理解——FSDP 的 forward 和 backward 都隐式触发 collective 操作。

### 3.3 Zero-Attention Mask 修复

**代码位置：** `verl/workers/actor/dp_opsd_actor.py:774-777`

```python
if torch.sum(dummy_teacher_mb["attention_mask"]) <= 0:
    dummy_teacher_mb["attention_mask"][:, -1] = 1
```

**新学到的点：**

`unpad_input()` 函数（用于 padding-free training）在处理全零 attention mask 时会 crash，因为它无法找到任何有效 token。

解决方案：强制设置最后一个 token 为有效，确保 `unpad_input()` 至少有一个 token 可以处理。

**面试价值：** 这是一个典型的"edge case 处理"——只有在分布式训练 + padding-free + dummy data 的组合下才会出现。

---

## 4. 训练策略的工程考量

### 4.1 Lambda Schedule 的设计逻辑

**代码位置：** `verl/trainer/opsd_trainer.py:1952-1961`

**新学到的点：**

RLSD 的 lambda schedule 有三个阶段：
1. **Warmup（0 → target）：** 训练初期，教师信号可能太强（Delta_t 大），需要慢慢引入
2. **Hold（target）：** 稳定训练阶段
3. **Decay（target → 0）：** 训练后期，学生接近教师，Delta_t 自然衰减，手动加速退场

**实际影响：** 在 100 步的快速验证中，`lambda_decay_steps=30` 意味着：
- Step 0-30: lambda 从 0.5 线性衰减到 0
- Step 30-100: 纯 GRPO（lambda=0）

这比原始配置的 60 步衰减更快，因为总步数更少。

**面试价值：** 展示了对 curriculum learning 在 RLSD 中的理解。

### 4.2 Teacher Sync 的工程实现

**代码位置：** `verl/workers/actor/dp_opsd_actor.py:69-75`

```python
def _sync_teacher_from_actor(self):
    if self.teacher_module is None or self.teacher_module is self.actor_module:
        return
    with torch.no_grad():
        for t_param, a_param in zip(self.teacher_module.parameters(), self.actor_module.parameters()):
            t_param.data.copy_(a_param.data)
```

**新学到的点：**

Teacher sync 是一个简单的 `data.copy_()` 操作，但有几个工程细节：
1. 必须用 `torch.no_grad()` 包裹，避免 autograd 追踪
2. Teacher 和 actor 必须在相同设备上（都是 GPU 或都是 CPU）
3. 如果 teacher 是 frozen 的（`freeze_teacher_model=True`），sync 会覆盖 frozen 状态

**实际影响：** 在 3090 上，如果 teacher 和 actor 都在 GPU 上，sync 很快。但如果 teacher 被 offload 到 CPU，sync 需要 CPU → GPU → CPU 的数据传输。

### 4.3 Reward Function 的进程隔离

**代码位置：** `examples/visual_rl/reward_function/math_task.py:58-122`

**新学到的点：**

数学题的 symbolic grader（`mathruler`）在一个独立子进程中运行，有：
- 10 秒 timeout
- 1GB 内存限制（`RLIMIT_AS`）
- Daemon process + terminate/kill fallback

这是因为 `mathruler` 可能在 adversarial input 上 hang 或 OOM。如果在主进程中运行，会阻塞整个训练。

**面试价值：** 展示了"防御性编程"在训练系统中的重要性。

---

## 5. 模型选择与适配

### 5.1 Qwen3-VL-4B vs 8B 的权衡

**新学到的点：**

| 维度 | 4B | 8B |
|------|----|----|
| 参数量 | 4B | 8B |
| FSDP 每卡分片 | ~1 GB | ~2 GB |
| 训练每卡显存 | ~8-12 GB | ~12-18 GB |
| 可用 batch size | 32-64 | 8-16 |
| 可用序列长度 | 4096 | 2048 |
| 推理能力 | 较弱 | 较强 |
| RLSD 效果 | 未知 | 论文验证 |

**实际影响：** 4B 模型在 3090 上可以跑更合理的实验设置（更大 batch、更长序列），但 RLSD 的效果可能与论文不同。

**面试价值：** 展示了对"资源约束下的模型选择"的理解。

### 5.2 freeze_vision_tower 的影响

**新学到的点：**

Qwen3-VL 的 vision tower（视觉编码器）参数量约占总模型的 20-30%。冻结它可以：
1. 节省 ~20% 的梯度和优化器状态显存
2. 减少 ~20% 的 FSDP 通信量
3. 降低 catastrophic forgetting 的风险

但代价是：
1. 模型无法通过 RL 训练改进视觉理解能力
2. 对于需要精细视觉 grounding 的任务，可能效果差

**实际影响：** 在 3090 上，冻结 vision tower 是一个必要的显存优化。对于数学推理任务（主要依赖文本理解），影响不大。

### 5.3 `limit_images` 的作用

**新学到的点：**

多模态数据中，一张图像可能被编码为数百甚至数千个 token。`limit_images=2` 限制每条数据最多使用 2 张图像，可以：
1. 大幅减少 prompt 长度
2. 减少 vLLM 的 KV Cache 显存
3. 加速推理

**实际影响：** 对于需要多张图像的任务（如多图对比），这个限制会影响效果。但对于大部分数学题（单图或无图），影响不大。

---

## 6. 分布式训练的工程模式

### 6.1 Ray 的角色

**新学到的点：**

在 RLSD 中，Ray 不做模型并行，而是做 **任务调度**：
- Head node 运行训练脚本（driver）
- Worker node 运行 FSDP workers
- Ray 负责在 workers 之间分发任务（如 `generate_sequences`、`update_actor`）
- 每个任务内部用 FSDP 做模型并行

**实际影响：** 在单机 8 卡上，Ray 的调度开销很小（都在同一台机器上）。但 Ray 的进程管理有 overhead（如 gRPC 通信）。

### 6.2 Dispatch Mode

**新学到的点：**

代码中的 `@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)` 表示：
- `DP_COMPUTE_PROTO`：数据并行，每个 GPU 处理一部分数据，结果合并
- `ONE_TO_ALL`：所有 GPU 执行相同操作（如 checkpoint save）

这是 veRL 框架的分布式调度抽象。

### 6.3 模型合并（Checkpoint Merging）

**代码位置：** `scripts/model_merger.py`

**新学到的点：**

FSDP 训练后的 checkpoint 是分片的（每个 rank 一个 `model_world_size_X_rank_Y.pt`）。使用时需要合并：

1. 加载所有 shard
2. 识别 DTensor 的 placement（`Replicate` vs `Shard(dim)`）
3. `Replicate`：取第一个 shard
4. `Shard(dim)`：`torch.cat` 沿分片维度拼接
5. 保存为完整模型

**实际影响：** 训练结束后，需要用 `model_merger.py` 合并 checkpoint 才能用于推理或评估。

---

## 7. 数据工程

### 7.1 多模态数据的处理流程

**新学到的点：**

多模态数据从 JSONL 到模型输入的完整流程：

```
JSONL → Dataset → DataLoader → Prompt Template (Jinja2) → Tokenizer → Processor (image/video)
```

每个环节都有工程挑战：
- **Dataset：** 需要处理图像路径、视频路径、预处理等
- **Jinja2：** 根据 `problem_type` 路由到不同的 prompt 模板
- **Tokenizer：** 多模态 token 的特殊处理（如 `<image>` 占位符）
- **Processor：** 图像 resize、视频帧采样、pixel 值归一化

### 7.2 Sequence Length Balancing

**代码位置：** `verl/utils/seqlen_balancing.py`

**新学到的点：**

在分布式训练中，不同 GPU 处理不同长度的序列。如果某个 GPU 分到了特别长的序列，它会成为 straggler（拖慢整个训练）。

解决方案：使用 Karmarkar-Karp 算法（largest differencing method）将序列按长度均衡分配到各 GPU，使每个 GPU 的总 token 数接近。

**实际影响：** 在 3090 上，序列长度差异导致的显存不均衡更严重（总显存小），sequence length balancing 更重要。

---

## 8. 面试中可以深挖的新点

### 8.1 "如果让你在 8x 3090 上训练 8B 模型，你会怎么做？"

> 1. 开启 FSDP full shard + CPU offload（参数和优化器都 offload）
> 2. 使用 gradient checkpointing
> 3. 冻结 vision tower
> 4. 用最小 batch size（`micro_batch_size=1`），不用 gradient accumulation（因为与 CPU offload 冲突）
> 5. 减小序列长度（`max_prompt_length=2048, max_response_length=1024`）
> 6. 减小 vLLM 的 `gpu_memory_utilization` 到 0.3
> 7. 接受训练速度很慢（每步可能 10+ 分钟）

### 8.2 "分布式训练中，如何处理不同 GPU 上有效样本数不同的问题？"

> 1. `all_reduce(MIN)` 确保所有 rank 的 batch size 一致
> 2. Dummy forward-backward（`loss * 0.0`）保持 FSDP collective 同步
> 3. Zero-attention mask 修复防止 `unpad_input` crash
> 4. Loss scaling 使用全局 token 数（`all_reduce(SUM)`）而非局部 token 数

### 8.3 "vLLM 和 FSDP 如何在同一 GPU 上共存？"

> 分时复用：
> 1. Rollout 阶段：vLLM 加载到 GPU，FSDP 模型 offload 到 CPU
> 2. Training 阶段：vLLM offload（释放显存），FSDP 模型加载到 GPU
> 3. 通过 `prepare_rollout_engine()` 和 `release_rollout_engine()` 管理生命周期
> 4. `gpu_memory_utilization` 控制 vLLM 的显存上限，留够 FSDP 训练用的空间

### 8.4 "CUDA 显存碎片是怎么产生的？怎么缓解？"

> 产生：频繁分配/释放不同大小的 tensor，导致显存中出现大量不连续的空闲块。
>
> 缓解：
> 1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — 使用可扩展的 segment
> 2. `torch.cuda.empty_cache()` — 释放未使用的缓存
> 3. 避免频繁的 tensor 创建/销毁（如用 in-place 操作）
> 4. 在 24GB 的 3090 上，碎片问题更严重，需要更积极的显存管理
