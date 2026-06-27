# RLSD 8x RTX 3090 复现计划

> 目标：在单机 8 卡 RTX 3090（24GB/卡，192GB 总显存）上完整复现 RLSD 的 GRPO baseline + RLSD 训练流程。

---

## 1. 硬件资源评估

### 1.1 原始配置 vs 你的硬件

| 维度 | 原始配置（4x8 H200） | 你的硬件（8x 3090） | 差距 |
|------|---------------------|---------------------|------|
| GPU 数量 | 32 | 8 | 4x |
| 单卡显存 | 140GB | 24GB | 5.8x |
| 总显存 | 4480GB | 192GB | 23x |
| GPU 架构 | Hopper (BF16 原生) | Ampere (BF16 需软件支持) | — |
| NVLink | 有 | 无（3090 只有 PCIe） | — |
| 互联带宽 | 900GB/s NVLink | 64GB/s PCIe 4.0 | 14x |

### 1.2 显存预算分析（单卡 24GB）

**RLSD 训练时的显存占用来源：**

| 组件 | FP16/BF16 估算 | 说明 |
|------|---------------|------|
| **Actor 模型参数（8B）** | ~16 GB | FSDP 分片后每卡 ~2GB |
| **AdamW 优化器状态** | ~32 GB | 2x FP32 momentum/variance，FSDP 分片后每卡 ~4GB |
| **梯度** | ~16 GB | FSDP 分片后每卡 ~2GB |
| **激活值（forward）** | 5-15 GB | 取决于序列长度，gradient checkpointing 可降至 ~3GB |
| **vLLM KV Cache** | 可配 | `gpu_memory_utilization` 控制 |
| **Teacher forward** | ~2GB | 与 actor 共享参数，只需额外 logits |

**训练阶段单卡峰值估算（FSDP full shard + CPU offload）：**

```
模型分片:      ~2 GB
优化器分片:    ~4 GB
梯度分片:      ~2 GB
激活值:        ~3-5 GB (gradient checkpointing)
Teacher logits: ~1-2 GB
总计:          ~12-15 GB
```

**Rollout 阶段单卡显存：**

```
vLLM 引擎:     ~8-12 GB (取决于 gpu_memory_utilization)
KV Cache:      剩余显存
```

### 1.3 关键约束

| 约束 | 影响 | 解决方案 |
|------|------|---------|
| 24GB 显存 | 无法同时放模型训练 + vLLM rollout | 分时复用：rollout 后 offload vLLM，再加载训练模型 |
| FSDP CPU offload 不能与 gradient accumulation 共存 | 必须 `micro_batch_size == global_batch_size_per_device` | 减小 global batch size |
| 无 NVLink | PCIe 通信慢，FSDP all-gather/reduce-scatter 有开销 | 减小通信量（full shard） |
| 3090 BF16 性能不如 A100/H100 | 训练速度慢 ~2-3x | 接受更长训练时间 |
| 无 NVLink | tensor_parallel 不划算 | `tensor_parallel_size=1` |

---

## 2. 适配策略总览

### 2.1 三条路径

| 路径 | 方案 | 可行性 | 推荐度 |
|------|------|--------|--------|
| **A: 8B 模型 + 极限压缩** | Qwen3-VL-8B + 全部 CPU offload + 极小 batch | 可行但极慢 | ⭐⭐ |
| **B: 4B 模型（推荐）** | Qwen3-VL-4B + 适度压缩 | 可行且合理 | ⭐⭐⭐⭐⭐ |
| **C: 8B 模型 + LoRA** | Qwen3-VL-8B + LoRA r=8 | 可行但改变了方法 | ⭐⭐⭐ |

**推荐路径 B 的理由：**
1. 4B 模型 FSDP 后每卡仅 ~1GB 参数分片，训练显存充裕
2. 可以保留较大的 batch size 和序列长度
3. RLSD 的核心贡献是算法设计，不依赖特定模型规模
4. 论文中 Qwen3-VL-4B-Thinking 用于数据筛选，说明 4B 模型有足够推理能力
5. 训练速度合理，可以跑完完整实验

---

## 3. 详细复现计划

### Phase 0: 环境搭建（Day 1）

#### 0.1 创建 Conda 环境

```bash
conda create -n rlsd python=3.11 -y
conda activate rlsd
```

#### 0.2 安装 PyTorch（3090 适配版）

```bash
# 3090 需要 CUDA 11.8 或 12.x
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

#### 0.3 安装 EasyVideoR1 基础环境

```bash
cd /home/chenyizhou
git clone https://github.com/cyuQ1n/EasyVideoR1.git
cd EasyVideoR1
pip install -e .
```

#### 0.4 安装 flash-attn（3090 兼容版）

```bash
# 3090 (sm_86) 需要 flash-attn 2.x
pip install flash-attn==2.8.3 --no-build-isolation
```

> **注意：** 如果 flash-attn 编译失败，尝试：
> ```bash
> MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation
> # 或者使用预编译 wheel
> pip install flash-attn --no-build-isolation
> ```

#### 0.5 安装 RLSD 项目

```bash
cd /home/chenyizhou/RLSD
pip install -e .
```

#### 0.6 验证安装

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "import vllm; print(vllm.__version__)"
python -c "import verl; print('verl ok')"
```

---

### Phase 1: 数据准备（Day 1-2）

#### 1.1 下载训练数据

```bash
# 从 HuggingFace 下载
cd /home/chenyizhou/RLSD
mkdir -p data

# 方法 1: 使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download iieycx/rlsd-train-MMFineReason-123K --repo-type dataset --local-dir data/raw

# 方法 2: 使用 git lfs
git lfs install
git clone https://huggingface.co/datasets/iieycx/rlsd-train-MMFineReason-123K data/raw
```

#### 1.2 准备训练/验证集

```bash
# 检查数据格式
head -n 2 data/raw/train.jsonl

# 按项目要求准备数据
# 需要的字段: problem, answer, problem_type, data_type, images, videos
# 对于纯文本数学题: data_type="text", images=[], videos=[]
```

#### 1.3 筛选适合 3090 的子集（可选但推荐）

由于 3090 显存有限，建议先用 **纯文本数学题**（无图像/视频）做初步验证：

```bash
# 筛选 data_type="text" 的样本
python -c "
import json
with open('data/raw/train.jsonl') as f:
    data = [json.loads(l) for l in f]
text_data = [d for d in data if d.get('data_type') == 'text']
print(f'Total: {len(data)}, Text-only: {len(text_data)}')

# 如果文本数据太少，用全部数据但限制图像分辨率
with open('data/train.jsonl', 'w') as f:
    for d in data[:10000]:  # 先用 10K 样本验证
        f.write(json.dumps(d) + '\n')
"
```

#### 1.4 准备验证集

```bash
# 从训练集中分出一小部分作为验证
python -c "
import json
with open('data/train.jsonl') as f:
    data = [json.loads(l) for l in f]
val_data = data[:500]  # 500 条验证
train_data = data[500:]
with open('data/train.jsonl', 'w') as f:
    for d in train_data:
        f.write(json.dumps(d) + '\n')
with open('data/val.jsonl', 'w') as f:
    for d in val_data:
        f.write(json.dumps(d) + '\n')
print(f'Train: {len(train_data)}, Val: {len(val_data)}')
"
```

---

### Phase 2: GRPO Baseline（Day 2-4）

> 先跑通 GRPO baseline，验证环境和流程正确，再上 RLSD。

#### 2.1 下载 Qwen3-VL-4B 模型

```bash
# 方法 1: huggingface-cli
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir /home/chenyizhou/models/Qwen3-VL-4B-Instruct

# 方法 2: 在代码中直接用 HuggingFace 名称（会自动下载）
```

#### 2.2 创建 3090 适配的 GRPO 配置

创建 `examples/visual_rl/grpo_3090_config.yaml`：

```yaml
data:
  train_files: data/train.jsonl
  val_files: data/val.jsonl

  prompt_key: problem
  answer_key: answer
  image_key: images
  video_key: videos

  image_dir: null
  video_fps: 2.0
  video_max_frames: 64          # 减少帧数
  use_preprocessed_videos: false

  min_pixels: 262144
  image_max_pixels: 524288      # 从 1048576 减半
  video_max_pixels: 131072      # 从 262144 减半

  max_prompt_length: 4096       # 从 16384 大幅缩减
  max_response_length: 2048     # 从 4096 减半
  filter_overlong_prompts: true # 开启过滤

  rollout_batch_size: 32        # 从 256 大幅缩减
  mini_rollout_batch_size: 8    # 分小批生成，避免 OOM
  val_batch_size: 16

  format_prompt: examples/visual_rl/format_prompt/unified.jinja
  override_chat_template: null
  shuffle: true                 # 小数据集建议开启 shuffle
  seed: 1

algorithm:
  adv_estimator: grpo
  disable_kl: true
  use_kl_loss: false
  kl_penalty: low_var_kl
  kl_coef: 0.01

worker:
  actor:
    global_batch_size: 32       # 从 256 大幅缩减
    micro_batch_size_per_device_for_update: 1
    micro_batch_size_per_device_for_experience: 1
    max_grad_norm: 1.0
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28

    padding_free: true
    dynamic_batching: true
    ulysses_size: 1

    model:
      model_path: /home/chenyizhou/models/Qwen3-VL-4B-Instruct  # 4B 模型
      enable_gradient_checkpointing: true
      trust_remote_code: false
      freeze_vision_tower: true  # 冻结视觉编码器节省显存

    optim:
      lr: 1.0e-6
      weight_decay: 0.01
      strategy: adamw
      lr_warmup_ratio: 0.0

    fsdp:
      enable_full_shard: true
      enable_cpu_offload: false   # 手动 offload 更灵活
      enable_rank0_init: true

    offload:
      offload_params: true        # 开启参数 CPU offload
      offload_optimizer: true     # 开启优化器 CPU offload

  rollout:
    n: 4                          # 从 8 减到 4
    temperature: 1.0
    top_p: 1.0

    limit_images: 2               # 限制每条数据最多 2 张图
    gpu_memory_utilization: 0.4   # 从 0.6 降到 0.4
    enforce_eager: false
    enable_chunked_prefill: true
    tensor_parallel_size: 1       # 3090 无 NVLink，不用 TP
    max_num_batched_tokens: 8192  # 从 32768 减少
    disable_tqdm: false

    val_override_config:
      temperature: 0.7
      top_p: 0.8
      top_k: 20
      repetition_penalty: 1.0
      presence_penalty: 1.5
      n: 1

  ref:
    fsdp:
      enable_full_shard: true
      enable_cpu_offload: true    # Ref 模型必须 CPU offload
      enable_rank0_init: true
    offload:
      offload_params: false

  reward:
    reward_function: examples/visual_rl/reward_function/unified.py:compute_score

trainer:
  total_epochs: 3                 # 先跑 3 个 epoch 验证
  max_steps: 100                  # 限制 100 步快速验证

  project_name: rlsd_3090
  experiment_name: grpo_4b_baseline

  logger: ["file"]

  nnodes: 1
  n_gpus_per_node: 8

  max_try_make_batch: 20
  val_freq: 10                    # 每 10 步验证
  val_before_train: true
  val_only: false
  val_generations_to_log: 3

  save_freq: 20
  save_limit: 3
  save_model_only: false
  save_checkpoint_path: null
  load_checkpoint_path: null
  find_last_checkpoint: true
```

#### 2.3 创建 3090 训练脚本

创建 `examples/visual_rl/grpo_3090_train.sh`：

```bash
#!/bin/bash
set -euo pipefail
set -x

export TOKENIZERS_PARALLELISM=false
export RAY_worker_num_grpc_internal_threads=1
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800000
export NCCL_DEBUG=WARN
export VERL_LOG_LEVEL=INFO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="/home/chenyizhou/models/Qwen3-VL-4B-Instruct"
CONFIG_PATH="${PROJECT_DIR}/examples/visual_rl/grpo_3090_config.yaml"
TRAIN_DATA="${PROJECT_DIR}/data/train.jsonl"
VAL_DATA="${PROJECT_DIR}/data/val.jsonl"
FORMAT_PROMPT="${PROJECT_DIR}/examples/visual_rl/format_prompt/unified.jinja"
REWARD_FUNCTION="${PROJECT_DIR}/examples/visual_rl/reward_function/unified.py:compute_score"

OUTPUT_PATH="${PROJECT_DIR}/checkpoints/grpo_3090"
EXPERIMENT_NAME="grpo_4b_baseline"
mkdir -p "${OUTPUT_PATH}/${EXPERIMENT_NAME}"

# 清理旧 Ray 进程
ray stop --force 2>/dev/null || true
sleep 3

# 启动 Ray head
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=8 \
    --disable-usage-stats

# 启动训练
python3 -m verl.trainer.main \
    config=${CONFIG_PATH} \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.format_prompt=${FORMAT_PROMPT} \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.rollout_batch_size=32 \
    data.mini_rollout_batch_size=8 \
    worker.rollout.n=4 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=32 \
    worker.actor.model.freeze_vision_tower=True \
    worker.actor.offload.offload_params=True \
    worker.actor.offload.offload_optimizer=True \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    trainer.project_name=rlsd_3090 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.save_checkpoint_path="${OUTPUT_PATH}/${EXPERIMENT_NAME}" \
    trainer.max_steps=100 \
    trainer.val_freq=10 \
    trainer.save_freq=20 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    2>&1 | tee "${OUTPUT_PATH}/train.log"

ray stop --force 2>/dev/null || true
echo "GRPO baseline training completed!"
```

#### 2.4 运行 GRPO Baseline

```bash
cd /home/chenyizhou/RLSD
bash examples/visual_rl/grpo_3090_train.sh
```

#### 2.5 预期问题与排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| CUDA OOM | batch size 太大 | 减小 `rollout_batch_size` 或 `mini_rollout_batch_size` |
| vLLM OOM | KV cache 太大 | 减小 `gpu_memory_utilization` 到 0.3 |
| FSDP OOM | 模型太大 | 确认 `offload_params=true`, `offload_optimizer=true` |
| NCCL timeout | PCIe 通信慢 | 增加 `NCCL_TIMEOUT` |
| flash-attn 错误 | 3090 兼容性 | 降级 flash-attn 版本或禁用 |
| 数据加载错误 | 路径或格式问题 | 检查 `data/train.jsonl` 格式 |

---

### Phase 3: RLSD 训练（Day 4-7）

> 在 GRPO baseline 跑通后，切换到 RLSD。

#### 3.1 创建 3090 适配的 RLSD 配置

创建 `examples/visual_rl/rlsd_3090_config.yaml`：

```yaml
data:
  train_files: data/train.jsonl
  val_files: data/val.jsonl

  prompt_key: problem
  answer_key: answer
  image_key: images
  video_key: videos
  trajectory_key: answer         # 用 answer 作为 trajectory（简化版）

  image_dir: null
  video_fps: 2.0
  video_max_frames: 64
  use_preprocessed_videos: false

  min_pixels: 262144
  image_max_pixels: 524288
  video_max_pixels: 131072

  max_prompt_length: 4096
  max_response_length: 2048
  filter_overlong_prompts: true

  rollout_batch_size: 32
  mini_rollout_batch_size: 8
  val_batch_size: 16

  format_prompt: examples/visual_rl/format_prompt/unified.jinja
  override_chat_template: null
  shuffle: true
  seed: 1

algorithm:
  adv_estimator: grpo
  disable_kl: true
  use_kl_loss: false
  kl_penalty: low_var_kl
  kl_coef: 0.01

opsd:
  use_gt_as_hint: true                           # 用 GT answer 作为 hint
  teacher_hint_template: examples/visual_rl/format_prompt/teacher_hint_self_distill.jinja
  freeze_teacher_model: true                     # 冻结 teacher
  enable_grpo_opsd_simple: true                  # 简化 RLSD 模式
  rlsd_lambda: 0.5
  rlsd_reweight_clip_range: 0.2
  rlsd_lambda_warmup_steps: 0
  rlsd_lambda_decay_steps: 30                    # 30 步内衰减到 0
  rlsd_teacher_sync_interval: 10                 # 每 10 步同步 teacher

worker:
  actor:
    global_batch_size: 32
    micro_batch_size_per_device_for_update: 1
    micro_batch_size_per_device_for_experience: 1
    max_grad_norm: 1.0
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28

    padding_free: true
    dynamic_batching: true
    ulysses_size: 1

    model:
      model_path: /home/chenyizhou/models/Qwen3-VL-4B-Instruct
      enable_gradient_checkpointing: true
      trust_remote_code: false
      freeze_vision_tower: true

    optim:
      lr: 1.0e-6
      weight_decay: 0.01
      strategy: adamw
      lr_warmup_ratio: 0.0

    fsdp:
      enable_full_shard: true
      enable_cpu_offload: false
      enable_rank0_init: true

    offload:
      offload_params: true
      offload_optimizer: true

  rollout:
    n: 4
    temperature: 1.0
    top_p: 1.0

    limit_images: 2
    gpu_memory_utilization: 0.4
    enforce_eager: false
    enable_chunked_prefill: true
    tensor_parallel_size: 1
    max_num_batched_tokens: 8192
    disable_tqdm: false

    val_override_config:
      temperature: 0.7
      top_p: 0.8
      top_k: 20
      repetition_penalty: 1.0
      presence_penalty: 1.5
      n: 1

  ref:
    fsdp:
      enable_full_shard: true
      enable_cpu_offload: true
      enable_rank0_init: true
    offload:
      offload_params: false

  reward:
    reward_function: examples/visual_rl/reward_function/unified.py:compute_score

trainer:
  total_epochs: 3
  max_steps: 100

  project_name: rlsd_3090
  experiment_name: rlsd_4b

  logger: ["file"]

  nnodes: 1
  n_gpus_per_node: 8

  max_try_make_batch: 20
  val_freq: 10
  val_before_train: true
  val_only: false
  val_generations_to_log: 3

  save_freq: 20
  save_limit: 3
  save_model_only: false
  save_checkpoint_path: null
  load_checkpoint_path: null
  find_last_checkpoint: true
```

#### 3.2 创建 RLSD 训练脚本

创建 `examples/visual_rl/rlsd_3090_train.sh`：

```bash
#!/bin/bash
set -euo pipefail
set -x

export TOKENIZERS_PARALLELISM=false
export RAY_worker_num_grpc_internal_threads=1
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800000
export NCCL_DEBUG=WARN
export VERL_LOG_LEVEL=INFO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="/home/chenyizhou/models/Qwen3-VL-4B-Instruct"
CONFIG_PATH="${PROJECT_DIR}/examples/visual_rl/rlsd_3090_config.yaml"
TRAIN_DATA="${PROJECT_DIR}/data/train.jsonl"
VAL_DATA="${PROJECT_DIR}/data/val.jsonl"
FORMAT_PROMPT="${PROJECT_DIR}/examples/visual_rl/format_prompt/unified.jinja"
TEACHER_TEMPLATE="${PROJECT_DIR}/examples/visual_rl/format_prompt/teacher_hint_self_distill.jinja"
REWARD_FUNCTION="${PROJECT_DIR}/examples/visual_rl/reward_function/unified.py:compute_score"

OUTPUT_PATH="${PROJECT_DIR}/checkpoints/rlsd_3090"
EXPERIMENT_NAME="rlsd_4b"
mkdir -p "${OUTPUT_PATH}/${EXPERIMENT_NAME}"

ray stop --force 2>/dev/null || true
sleep 3

ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=8 \
    --disable-usage-stats

python3 -m verl.trainer.opsd_main \
    config=${CONFIG_PATH} \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.format_prompt=${FORMAT_PROMPT} \
    data.trajectory_key=answer \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.rollout_batch_size=32 \
    data.mini_rollout_batch_size=8 \
    worker.rollout.n=4 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=32 \
    worker.actor.model.freeze_vision_tower=True \
    worker.actor.offload.offload_params=True \
    worker.actor.offload.offload_optimizer=True \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    algorithm.adv_estimator=grpo \
    algorithm.disable_kl=True \
    opsd.use_gt_as_hint=True \
    opsd.teacher_hint_template=${TEACHER_TEMPLATE} \
    opsd.enable_grpo_opsd_simple=True \
    opsd.freeze_teacher_model=True \
    opsd.rlsd_lambda=0.5 \
    opsd.rlsd_lambda_warmup_steps=0 \
    opsd.rlsd_lambda_decay_steps=30 \
    opsd.rlsd_reweight_clip_range=0.2 \
    opsd.rlsd_teacher_sync_interval=10 \
    trainer.project_name=rlsd_3090 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.save_checkpoint_path="${OUTPUT_PATH}/${EXPERIMENT_NAME}" \
    trainer.max_steps=100 \
    trainer.val_freq=10 \
    trainer.save_freq=20 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    2>&1 | tee "${OUTPUT_PATH}/train.log"

ray stop --force 2>/dev/null || true
echo "RLSD training completed!"
```

#### 3.3 运行 RLSD 训练

```bash
cd /home/chenyizhou/RLSD
bash examples/visual_rl/rlsd_3090_train.sh
```

---

### Phase 4: 对比分析（Day 7-8）

#### 4.1 对比 GRPO vs RLSD

```bash
# 对比两个实验的训练日志
# 关注: reward 曲线、entropy、clip ratio、STGCA metrics
```

#### 4.2 评估指标

在验证集上对比：
- Accuracy（答案正确率）
- Format compliance（格式遵循率）
- Response length（响应长度分布）

#### 4.3 可选：更多实验

- **Ablation lambda：** 对比 `rlsd_lambda=0.3/0.5/0.7`
- **Ablation teacher sync：** 对比 `teacher_sync_interval=5/10/20`
- **Ablation clip range：** 对比 `rlsd_reweight_clip_range=0.1/0.2/0.3`

---

## 4. 关键参数对照表

| 参数 | 原始值（H200） | 3090 适配值 | 说明 |
|------|---------------|------------|------|
| Model | Qwen3-VL-8B | **Qwen3-VL-4B** | 显存限制 |
| rollout_batch_size | 256 | **32** | 减少 8x |
| mini_rollout_batch_size | null | **8** | 分批生成 |
| global_batch_size | 256 | **32** | 减少 8x |
| rollout.n | 8 | **4** | 减少 2x |
| max_prompt_length | 16384 | **4096** | 减少 4x |
| max_response_length | 4096 | **2048** | 减少 2x |
| gpu_memory_utilization | 0.6 | **0.4** | 减少 vLLM 显存 |
| max_num_batched_tokens | 32768 | **8192** | 减少 4x |
| limit_images | 0 | **2** | 限制图像数 |
| image_max_pixels | 1048576 | **524288** | 减半 |
| freeze_vision_tower | false | **true** | 节省显存 |
| offload_params | true | **true** | 保持 |
| offload_optimizer | true | **true** | 保持 |
| ref.enable_cpu_offload | true | **true** | 保持 |
| rlsd_lambda_decay_steps | 60 | **30** | 步数少，衰减快 |
| total_epochs | 10 | **3** | 快速验证 |
| max_steps | null | **100** | 快速验证 |

---

## 5. 学习检查点

每个 Phase 完成后，你应该能回答以下问题：

### Phase 0 完成后
- [ ] FSDP 的 sharding strategy 有哪几种？区别是什么？
- [ ] CPU offload 的 `offload_params` 和 `offload_optimizer` 分别做什么？
- [ ] 为什么 FSDP CPU offload 不能与 gradient accumulation 共存？
- [ ] `expandable_segments:True` 解决什么问题？

### Phase 1 完成后
- [ ] RLSD 的训练数据格式是什么？需要哪些字段？
- [ ] `use_gt_as_hint=True` 时，Navigator 模块的作用是什么？
- [ ] `trajectory_key` 字段的作用是什么？

### Phase 2 完成后
- [ ] GRPO 的 advantage 计算流程是什么？
- [ ] `rollout.n` 和 `global_batch_size` 的关系是什么？
- [ ] 为什么 `disable_kl=True`？KL penalty 在这里的作用是什么？
- [ ] vLLM 的 `gpu_memory_utilization` 如何影响训练？

### Phase 3 完成后
- [ ] RLSD 的 `_build_stgca_advantages` 函数做了什么？
- [ ] `rlsd_lambda` 的 warmup + decay schedule 是怎么实现的？
- [ ] Teacher sync 是怎么工作的？为什么不同步更新？
- [ ] RLSD 和 GRPO 的训练动态有什么区别？

### Phase 4 完成后
- [ ] RLSD 在哪些指标上优于 GRPO？差距有多大？
- [ ] STGCA metrics（delta_mean, weight_mean, clip ratio）的变化趋势是什么？
- [ ] 如果继续做，你会怎么改进 RLSD？

---

## 6. 常见问题 FAQ

### Q1: 为什么推荐 4B 而不是 8B？

> 8B 模型在 24GB GPU 上训练需要非常激进的 CPU offload，导致：
> 1. 训练速度极慢（每步可能需要 10+ 分钟）
> 2. Batch size 极小（可能只能 1-2），GRPO 的组内标准化效果差
> 3. 序列长度受限，无法处理长推理
> 
> 4B 模型可以在保持合理速度的同时，保留足够的 batch size 和序列长度。

### Q2: 3090 不支持 BF16 怎么办？

> RTX 3090 (Ampere) 实际上支持 BF16 的软件模拟（通过 PyTorch），但没有硬件加速。训练中使用 FP16 更高效。FSDP 的 `mp_param_dtype` 默认会根据模型自动选择，不需要手动设置。

### Q3: 如果 8 卡 3090 还是 OOM 怎么办？

> 进一步压缩：
> 1. `rollout_batch_size` 从 32 降到 16
> 2. `rollout.n` 从 4 降到 2（但 GRPO 至少需要 2）
> 3. `max_response_length` 从 2048 降到 1024
> 4. `gpu_memory_utilization` 从 0.4 降到 0.3
> 5. 考虑使用 Qwen3-VL-2B（如果有的话）

### Q4: 训练速度大概多快？

> 预估（4B 模型 + 8x 3090）：
> - GRPO 每步: ~2-4 分钟
> - RLSD 每步: ~3-5 分钟（额外 teacher forward）
> - 100 步 GRPO: ~3-7 小时
> - 100 步 RLSD: ~5-8 小时
> 
> 完整 400 步实验: ~1-2 天

### Q5: 能不能用 LoRA 来适配 8B 模型？

> 可以，但会改变实验设置：
> 1. LoRA 只训练低秩增量，不更新全量参数
> 2. RLSD 的 teacher forward 需要完整的 log-probability，LoRA 不影响
> 3. 但 FSDP + LoRA 的组合需要额外配置
> 4. 实验结果与论文不完全可比
> 
> 如果一定要用 8B，LoRA 是一个可行的折中方案。
