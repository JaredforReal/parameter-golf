# train_gpt.py 与 train_gpt_mlx.py 架构详解

本文档详细解析这两个训练脚本的架构、设计理念和核心组件。

---

## 总体设计理念

两个脚本都遵循同一套核心设计原则，以在 **16MB 参数限制** 和 **10 分钟训练时间** 内优化模型性能：

1. **参数紧凑**：9 层 Transformer，512 维，1024 词表，共约 52 万参数
2. **多精度策略**：模型运行在 bfloat16，权重保持 float32，控制参数固定 float32
3. **优化器分组**：矩阵参数用 Muon，嵌入和标量用 Adam
4. **评估创新**：支持测试时 LoRA 自适应和 tokenizer 无关 BPB 指标

---

## 一、超参数配置（Hyperparameters）

### 数据和路径

```python
DATA_PATH           # 数据集根目录，默认 fineweb10B_sp1024
TOKENIZER_PATH      # sentencepiece tokenizer 模型
```

### 模型架构参数

| 参数          | train_gpt.py | train_gpt_mlx.py | 说明                |
| ------------- | ------------ | ---------------- | ------------------- |
| NUM_LAYERS    | 9            | 9                | 编码器+解码器总层数 |
| MODEL_DIM     | 512          | 512              | 隐藏维度            |
| NUM_HEADS     | 8            | 8                | 注意力头数          |
| NUM_KV_HEADS  | 4            | 4                | 键值头数（GQA）     |
| MLP_MULT      | 2            | 2                | MLP 中间层倍数      |
| VOCAB_SIZE    | 1024         | 1024             | 词表大小            |
| TRAIN_SEQ_LEN | 1024         | 1024             | 训练序列长度        |

### 训练参数

| 参数                  | 默认值  | 说明                   |
| --------------------- | ------- | ---------------------- |
| TRAIN_BATCH_TOKENS    | 524,288 | 每步训练的总 tokens 数 |
| ITERATIONS            | 20,000  | 最大训练步数           |
| WARMUP_STEPS          | 20      | 预热步数               |
| WARMDOWN_ITERS        | 1,200   | 最后冷却的步数         |
| MAX_WALLCLOCK_SECONDS | 600.0   | 10 分钟限制            |

### 优化器参数

#### Muon（矩阵参数）

```python
MATRIX_LR = 0.04                    # 矩阵学习率
MUON_MOMENTUM = 0.95                # 动量系数
MUON_BACKEND_STEPS = 5              # 正交化迭代次数
MUON_MOMENTUM_WARMUP_START = 0.85   # 预热起始动量
MUON_MOMENTUM_WARMUP_STEPS = 500    # 预热步数
```

#### Adam（嵌入和标量）

```python
TIED_EMBED_LR = 0.05    # 嵌入学习率
SCALAR_LR = 0.04        # 标量学习率
BETA1 = 0.9, BETA2 = 0.95, ADAM_EPS = 1e-8
```

### 测试时训练（LoRA）参数

```python
TTT_LORA_RANK = 8           # LoRA 秩
TTT_LORA_LR = 0.01          # LoRA 学习率
TTT_CHUNK_SIZE = 256        # 文档处理分块大小
TTT_EVAL_SEQ_LEN = 1024     # 评估序列长度
TTT_BATCH_SIZE = 64         # 批处理大小
```

---

## 二、Muon 优化器

### 核心思想

Muon 是一个**方向-动量优化器**，针对 LLM 训练优化：

1. **梯度正交化**：使用 Newton-Schulz 迭代将梯度投影到 Stiefel 流形
2. **动量在高维**：在正交化前应用动量，捕捉长期趋势
3. **缩放校正**：按矩阵宽高比缩放更新

### 实现细节

#### Newton-Schulz 迭代

```python
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10):
    # 将梯度 G 正交化为接近 Stiefel 流形上的点
    # 参考：https://kellerjordan.github.io/posts/muon/
    a, b, c = (3.4445, -4.7750, 2.0315)  # 收敛系数
    X = G.bfloat16() / (norm(G) + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X
```

#### 参数更新流程

1. 按 rank 和 world_size 分配参数到不同进程
2. 应用动量：`buf = momentum * buf + grad`
3. Nesterov 加速：`g_eff = grad + momentum * buf`
4. 正交化：`g_ortho = zeropower_via_newtonschulz5(g_eff)`
5. 缩放校正：`scale = sqrt(max(1, rows / cols))`
6. 参数更新：`p -= lr * g_ortho * scale`

### 分布式同步

```python
if distributed:
    dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
```

---

## 三、Tokenizer 无关评估（BPB 指标）

### 问题背景

- 嵌入参数 = $2 \times d_{model} \times d_{vocab}$，对小模型占比很大
- 不同 tokenizer 会产生完全不同的参数数量
- 需要一个 tokenizer 无关的评估指标

### BPB 计算流程

#### 1. 构建 SentencePiece 查找表

```python
def build_sentencepiece_luts(sp, vocab_size, device):
    base_bytes_lut        # 每个 token 的基础字节数
    has_leading_space_lut # 是否有前导空格（占 1 字节）
    is_boundary_token_lut # 是否边界 token（无字节成本）
```

#### 2. 验证阶段计算

```
val_loss = cross_entropy_loss  # 自然对数
bits_per_token = val_loss / ln(2)
tokens_per_byte = token_count / byte_count
val_bpb = bits_per_token * tokens_per_byte
```

#### 3. 关键逻辑

- 计算**每个目标 token 的字节成本**
- 若前一个 token 不是边界词，后续 token 的前导空格额外计 1 字节
- 最终 BPB = 压缩比的理论下界（假设最优编码）

---

## 四、模型架构

### 4.1 核心 Transformer 模块

#### RMSNorm

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        return x / sqrt(mean(x^2) + eps)
```

- 无学习参数版本，减少参数量
- 在 float32 中计算以保证稳定性

#### RoPE（旋转位置编码）

```python
def apply_rotary_emb(x, cos, sin):
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return [x1*cos + x2*sin, -x1*sin + x2*cos]
```

- 按头维进行旋转
- 使用余弦表缓存加速

#### CausalSelfAttention（因果自注意力）

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x):
        # 分离 Q/K/V 投影
        q = self.c_q(x)     # shape: [B, T, D]
        k = self.c_k(x)     # shape: [B, T, KV_dim]
        v = self.c_v(x)     # shape: [B, T, KV_dim]

        # RMSNorm + RoPE
        q = RoPE(RMSNorm(q))
        k = RoPE(RMSNorm(k))

        # 应用 Q gain（可学习的缩放）
        q = q * q_gain

        # GQA: 多个 Q 头共享一组 K/V
        attn = scaled_dot_product_attention(q, k, v, causal=True)
        return self.proj(attn)
```

**关键设计**：

- GQA（分组查询注意力）：4 个 KV 头对应 8 个 Q 头，减少 KV 缓存
- Q gain 参数：float32 保存，可独立调整查询头的学习率

#### MLP

```python
class MLP(nn.Module):
    def forward(self, x):
        x = relu(self.fc(x))
        return self.proj(x ** 2)  # relu^2 代替 GELU
```

#### Block（Transformer Block）

```python
class Block(nn.Module):
    def forward(self, x, x0, q_delta_fn=None, v_delta_fn=None):
        # 残差混合：动态混合原始输入和前层输出
        mix = self.resid_mix  # [2, D]
        x = mix[0] * x + mix[1] * x0

        # 注意力分支
        n = RMSNorm(x)
        x = x + attn_scale * attention(n)

        # MLP 分支
        x = x + mlp_scale * mlp(RMSNorm(x))

        return x
```

**关键创新**：

- `resid_mix`：学习的残差混合系数，可调整新旧信息比例
- `attn_scale` 和 `mlp_scale`：float32 标量，独立控制分支权重

### 4.2 GPT 模型

#### 编码器-解码器架构

```python
class GPT(nn.Module):
    def __init__(self, num_layers=9, ...):
        # 分裂：前 4 层为编码器，后 5 层为解码器
        self.num_encoder_layers = num_layers // 2      # 4
        self.num_decoder_layers = num_layers - 4       # 5

        # 跳过连接：编码器输出被解码器重新使用
        self.skip_weights = nn.Parameter(
            torch.ones(min(4, 5), model_dim)  # 4 个跳过权重
        )

    def forward(self, input_ids):
        x = embedding(input_ids)
        x = RMSNorm(x)
        x0 = x  # 保存原始嵌入用于残差

        # 编码器：保存每层输出
        skips = []
        for i in range(4):
            x = block[i](x, x0)
            skips.append(x)

        # 解码器：从后向前消费跳过连接
        for i in range(5):
            if skips:
                x = x + skip_weights[i] * skips.pop()  # 反向消费
            x = block[4 + i](x, x0)

        # 输出
        x = final_norm(x)
        logits = embedding(x)  # 绑定嵌入
        logits = logit_softcap * tanh(logits / logit_softcap)
```

**关键特点**：

1. **编码器-解码器跳跃**：编码器梯度直接流向解码器上采样层
2. **绑定嵌入**：输入和输出层共享权重，节省参数
3. **Logit Softcap**：限制输出范围 $[-c, c]$，防止梯度爆炸

---

## 五、后训练量化（INT8 + ZLIB）

### 量化策略

```
训练阶段：bfloat16/float32 混合精度
评估阶段：int8 量化 + zlib 压缩
```

### 量化格式：`int8_clean_per_row_v1`

#### 分类规则

| 张量类型 | 处理方式     | 说明                        |
| -------- | ------------ | --------------------------- |
| 2D 矩阵  | 按行 int8    | 每行一个缩放因子            |
| 1D/标量  | 按张量 int8  | 单个缩放因子                |
| 小张量   | fp16 传递    | numel ≤ 65536               |
| 控制参数 | float32 传递 | 如 attn_scale, mlp_scale 等 |
| 非浮点   | 直接传递     | 不量化                      |

#### 量化过程

```python
def quantize_float_tensor(t):
    if t.ndim == 2:
        # 按行量化
        clip_abs = quantile(|t|, 99.99984%)  # 每行
        clipped = clip(t, -clip_abs, clip_abs)
        scale = clip_abs / 127
        q = round(clipped / scale).astype(int8)
    else:
        # 按张量量化
        clip_abs = quantile(|t|.flatten(), 99.99984%)
        scale = clip_abs / 127
        q = round(clip(t, -clip_abs, clip_abs) / scale).astype(int8)

    return q, scale
```

#### 反量化过程

```python
def dequantize(q, scale):
    if scale.ndim > 0:  # 按行缩放
        return q.float() * scale[:, None]
    else:
        return q.float() * float(scale)
```

#### 最终工件大小

```
= len(code_bytes) + len(zlib.compress(quantized_state_dict))
≤ 16,000,000 bytes (十进制)
```

---

## 六、数据加载

### TokenStream（单进程流）

```python
class TokenStream:
    def __init__(self, pattern):
        # 全局共享数据流，无采样
        self.files = sorted(glob.glob(pattern))
        self.tokens = load_shard(self.files[0])
        self.pos = 0

    def take(self, n):
        # 连续读取，超出时自动切换文件
```

### DistributedTokenLoader（多进程分割）

```python
class DistributedTokenLoader:
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        # 全局：rank 0 读取 global_tokens * world_size 个 token
        # 本地：rank i 取 [i*span, (i+1)*span) 的切分
        # 确保每个 rank 看到不同的 token 前缀
```

**设计原则**：

- 确定性：无随机 shuffle，便于复现
- 分布式对齐：所有 rank 在冻结的 token 序列上训练
- 梯度累积友好：支持 grad_accum_steps

---

## 七、测试时 LoRA 适应（评估）

### 核心思想

在验证阶段为**每个文档独立训练低秩适配器**，允许模型快速适应文档特定语言特征。

### BatchedLinearLoRA

```python
class BatchedLinearLoRA(nn.Module):
    def __init__(self, bsz, in_features, out_features, rank):
        self.A = Parameter(bsz, rank, in_features)      # 下投影
        self.B = Parameter(bsz, out_features, rank)     # 上投影

    def forward(self, x):
        # x: [bsz, T, in]
        # (x @ A^T) @ B^T = x @ (BA)^T
        return (x @ A.T) @ B.T
```

### BatchedTTTLoRA

```python
class BatchedTTTLoRA:
    def __init__(self, bsz, model, rank):
        # LM Head 的 LoRA
        self.lm_head_lora = BatchedLinearLoRA(...)

        # 每个 block 的 Q 和 V LoRA
        self.q_loras = [BatchedLinearLoRA(...) for _ in blocks]
        self.v_loras = [BatchedLinearLoRA(...) for _ in blocks]
```

### 评估流程

#### 1. 文档切分

```python
docs = _find_docs(all_tokens)  # 查找 BOS 边界
```

#### 2. 文档分批处理

```python
for batch in chunks(docs, batch_size):
    lora.reset()  # 重置 LoRA 权重
    opt = Adam(lora.parameters())

    for chunk in doc_chunks:
        # 前向传播 + LoRA 适应
        loss = model(x, y, lora=lora)

        # LoRA 梯度步
        loss.backward()
        opt.step()
        opt.zero_grad()

    # 计算最终 BPB
    loss, bpb = evaluate_chunk(...)
```

#### 3. 滑动窗口评估

```python
def _compute_chunk_window(chunk_idx, pred_len, chunk_size, eval_seq_len):
    # 确保每个预测块都看到足够的上下文
    chunk_start = chunk_idx * chunk_size
    chunk_end = pred_len if is_last else (chunk_idx + 1) * chunk_size
    window_start = max(0, chunk_end - eval_seq_len)
    return (window_start, window_end, offset_in_window, ...)
```

---

## 八、train_gpt_mlx.py 特殊性

### MLX 框架特性

1. **统一内存**：Mac 的 CPU 和 GPU 共享内存
2. **惰性评估**：默认图构建不执行，需显式 `mx.eval()`
3. **自动求导**：无需显式反向传播

### MLX 特定优化

#### 微批处理和内存管理

```python
mlx_max_microbatch_tokens = 8192   # 每个微批的 token
mlx_eager_eval = True              # 强制每个微批后评估

# 子批循环中定期 eval
for chunk_tokens in chunk_sizes:
    x, y = train_loader.next_batch(chunk_tokens, seq_len)
    loss, grads = loss_and_grad_fn(x, y)
    grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
    if mlx_eager_eval:
        mx.eval(loss_accum, grad_accum)  # 物化图，释放内存
```

#### Muon 实现差异

```python
class Muon:
    def step(self, params, grads, step, lr_mul):
        # MLX 版本使用 tree_flatten/unflatten
        # 直接操作字典而非 nn.Module
```

#### 损失函数分块

```python
def loss_chunked(x, y):
    if logit_chunk_tokens <= 0:
        logits = x @ embedding.T
        return cross_entropy(logits, y)

    loss_sum = 0
    for s in range(0, n, logit_chunk_tokens):
        e = min(s + logit_chunk_tokens, n)
        logits = x[s:e] @ embedding.T
        loss_sum += cross_entropy(logits, y[s:e])
    return loss_sum / n
```

---

## 九、训练循环流程

### 预热阶段

```python
if warmup_steps > 0:
    # 1. 保存初始权重和优化器状态
    # 2. 运行 warmup_steps 步（激活编译路径）
    # 3. 恢复权重和优化器状态
    # 目的：预编译 CUDA 图，准确测量实际训练时间
```

### 主训练循环

```python
step = 0
t0 = perf_counter()

while True:
    # 梯度累积
    for accum_step in range(grad_accum_steps):
        x, y = train_loader.next_batch(...)
        loss = model(x, y) / grad_accum_steps
        loss.backward()

    # 学习率衰减
    lr_mul = compute_lr_mul(step, elapsed_ms)

    # 优化器步（分组）
    for opt in [opt_tok, opt_muon, opt_scalar]:
        opt.step(lr_mul=lr_mul)
    zero_grad()

    # 周期性验证
    if val_loss_every > 0 and step % val_loss_every == 0:
        val_loss, val_bpb = eval_val(...)

    # 检查时间和迭代数限制
    elapsed_ms = (perf_counter() - t0) * 1000
    if should_stop(step, elapsed_ms):
        break

    step += 1
```

### 学习率计划

```python
def lr_mul(step, elapsed_ms):
    if warmdown_iters <= 0:
        return 1.0

    if max_wallclock_seconds > 0:
        # 基于时间的衰减
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = warmdown_iters * step_ms
        remaining_ms = max(1000 * max_wallclock_seconds - elapsed_ms, 0)
        return remaining_ms / max(warmdown_ms, 1e-9)
    else:
        # 基于步数的衰减
        warmdown_start = max(iterations - warmdown_iters, 0)
        if warmdown_start <= step < iterations:
            return (iterations - step) / max(warmdown_iters, 1)
        return 1.0
```

---

## 十、序列化和验证

### 两阶段评估

#### 1. INT8 + ZLIB 评估

```python
# 保存压缩模型
quantized_state, stats = quantize_state_dict_int8(model.state_dict())
quantized_bytes = zlib.compress(pickle.dumps(quantized_state), level=9)

# 验证可复现性
loaded_state = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quantized_bytes)))
model.load_state_dict(loaded_state)
q_val_loss, q_val_bpb = eval_val(...)
```

#### 2. 测试时训练评估

```python
ttt_val_loss, ttt_val_bpb = eval_val_ttt_lora(...)  # 最终排行榜成绩
```

### 输出日志格式

```
final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:X.XXXX
final_int8_ttt_lora val_loss:X.XXXX val_bpb:X.XXXX  # 排行榜指标
```

---

## 总结

### 核心创新

1. **参数高效**：编码-解码跳跃、绑定嵌入、参数共享
2. **优化创新**：Muon + Adam 分组、梯度正交化
3. **评估科学**：Tokenizer 无关的 BPB、测试时 LoRA 适应
4. **量化智能**：按行 int8、控制参数保留 float32

### 性能特点

- **基线 BPB**：~1.224（baseline 在 10 分钟内）
- **SOTA BPB**：~1.175（使用高级优化和架构改进）
- **改进空间**：架构创新（滑动窗口、长上下文）、量化进阶（bitnet）、编码创新（自定义 tokenizer）
