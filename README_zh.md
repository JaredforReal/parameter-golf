<img width="3840" height="1280" alt="1920x640-discord" src="https://github.com/user-attachments/assets/90607b26-171f-476a-90ae-69b9dbb7cb30" />

<br>
<br>

**OpenAI Model Craft Challenge: Parameter Golf** 是一项挑战：在 8xH100 上用不超过 10 分钟训练出最优语言模型，且最终工件（artifact）需控制在 16MB 内。评估指标为 FineWeb 验证集上的压缩表现（与 tokenizer 无关，指标为 bits per byte）。

该挑战深受 [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) 启发。后者的目标是在最短时间内把 FineWeb 验证损失降到 3.28。我们很期待看到在“参数受限”设定下出现的新思路：例如独特架构（测试时计算、激进参数共享、深度递归、低秩训练等）、压缩方案（低精度、QAT、bitnet、全新 tokenizer 等），以及其他有创造力的提交（测试时训练、长上下文、megakernel 等）。

如果你熟悉 [神经网络缩放定律](https://arxiv.org/abs/2001.08361)，可以把这个挑战理解为一种 L(N) 优化：在固定参数量 N 下，尽可能优化最低损失，不限制数据、计算、训练步数或架构。像 [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt)（可视作 L(T)：在目标损失下最短时间）和 [NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun)（可视作 L(D)：在固定数据规模下最低损失）都可以看作同一类问题的不同版本。

理想情况下我们希望允许任意计算资源，但为避免挑战成本过高，**排行榜提交**限制为在 8xH100 上 10 分钟内完成。即便不满足该计算限制，我们也欢迎你在“非纪录提交”中分享结果：我们同样希望看到参数受限性能边界被不断推进。

我们也知道算力昂贵，因此 **OpenAI 提供 $1,000,000 的算力积分**帮助大家开始实验。可通过此表单申请：[Request a Compute Grant](https://openai.com/index/parameter-golf/#credit-form)。
申请时请注意选择合适的申请等级、提供充分理由，并且**使用与 OpenAI / ChatGPT 账号绑定的邮箱提交**。

## 参与者表单

如果你喜欢解决高难技术问题，欢迎填写 [Challenge Participant Form](https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf) 介绍自己。这样有助于我们给提交结果署名，并在未来联系你。_填写该表单并非参赛必需。_

OpenAI 的许多研究者最初都在顶级数学与编程竞赛中崭露头角。Model Craft Challenge 的设计初衷也类似：考察在陌生问题中展现创造力与严谨性的能力，而这些正是前沿 AI 研究的重要素质。

我们计划在 6 月招聘一小批早期研究人员，面向在读本科生与应届毕业生，包括奥赛奖牌获得者和顶级竞赛选手。对于表现突出的参与者，本挑战也可能成为在 OpenAI 研究者与招聘团队前脱颖而出的机会。

挑战时间：3 月 18 日至 4 月 30 日。

祝训练顺利！

## 排行榜

| Run                  |  Score | Author         | Summary                                         | Date       | Info                                                                                                |
| -------------------- | -----: | -------------- | ----------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------- |
| Muon WD + 10 layer   | 1.1748 | notapplica     | 包含此前改进 + 频谱嵌入初始化 + 残差混合        | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md) |
| Sliding Window Eval  | 1.1925 | Matthew Li     | 评估时采用 stride=64 的滑动窗口，逐步增大上下文 | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md)                             |
| Lora TTT             | 1.1928 | samacqua       | 使用 LoRA 的测试时训练（TTT）                   | 2026-03-19 | [info](records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md)                                      |
| 4k seq length        | 1.2014 | Spokane Way    | 4k 序列长度 + 更优超参数                        | 2026-03-19 | [info](records/track_10min_16mb/2026-03-18_LongContextSeq2048/README.md)                            |
| 2048 seq length      |  1.206 | Spokane Way    | 2048 序列长度（训练 + 验证）                    | 2026-03-18 | [info](records/track_10min_16mb/2026-03-18_LongContextSeq2048/README.md)                            |
| int6 mixed precision | 1.2147 | Nan Liu        | 10 层，int8/int6 混合精度                       | 2026-03-18 | [info](records/track_10min_16mb/2026-03-19_10L_MixedPrecision/README.md)                            |
| fp16 Embed           | 1.2197 | Renier Velazco | FP16 绑定嵌入 + LR/Warmdown 调优                | 2026-03-18 | [info](records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md)                              |
| Naive Baseline       | 1.2244 | Baseline       | 9 层 512 维 1024 词表，绑定嵌入，4 个 KV 头     | 2026-03-18 | [info](records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md)                                 |

#### 值得关注的非纪录运行

| Run             |  Score | Author     | Summary                              | Date       | Info                                                                                                 |
| --------------- | -----: | ---------- | ------------------------------------ | ---------- | ---------------------------------------------------------------------------------------------------- |
| 4-Hour Baseline | 1.2074 | Will DePue | 测试无限算力设定：8xH100 训练 4 小时 | 2026-03-18 | [info](records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md) |

## 快速开始

### 训练你的第一个模型（Apple Silicon Mac）

如果你使用 Apple Silicon 的 Mac（笔记本或台式机），我们提供了一个简洁的 MLX 训练脚本，方便本地快速迭代。

如果你没有 Apple Silicon Mac，也可以使用该脚本的无 MLX 版本。你可以让 [Codex](https://openai.com/codex/) 帮你重构，改动很直接。不过速度可能仍较慢，因此我们建议尽快切到云端 GPU（如 Runpod）。

首先，克隆仓库，创建新的 Python 环境，并安装 MLX 路径与数据下载所需包：

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
```

下载我们缓存好的 FineWeb 数据（1024-token 词表版本）：

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

这会生成 `./data/datasets/fineweb10B_sp1024/` 和 `./data/tokenizers/`。
默认会下载完整验证集 + 80 个训练分片（8B tokens）。如果你只想做本地小规模冒烟测试，可传入 `--train-shards 1`，例如 `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`。

然后运行一个小规模 MLX 训练任务：

```bash
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

验证始终使用完整 `fineweb_val_*` 切分（固定为前 5 万文档）。上面的冒烟命令会跳过周期性验证，仅在最后打印一次 `val_loss` 和 `val_bpb`。

### 扩展到远程机器

当你本地验证满意，或需要更多算力时，可切换到远程 CUDA 机器。

你可以从任何平台租 GPU，但 OpenAI 与 Runpod 合作，简化了部署流程。

#### 启动 1xH100 Pod

1. 先 [创建 Runpod 账号](https://console.runpod.io/deploy)。建议在左侧 Settings 中配置 SSH key，方便连接远程机器。如果你不熟悉流程，可以让 Codex 协助。

2. 完成账号配置后，创建新的 GPU Cloud Pod。你可选择任意 GPU 型号。排行榜最终提交必须在 8xH100（SXM 版本）上 10 分钟内完成，但强烈建议先在更便宜的机器上做实验，因为 8xH100 的成本可能在 ~$20/小时。

3. 先从 1xH100 Pod 开始。使用官方 Parameter Golf 模板部署：[Launch Template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th)。开启 SSH terminal access，其余保持默认。部署并 SSH 登录后，你会位于 `/workspace/`。

在远程机器上，把仓库克隆到本地磁盘。镜像中已预装所有 Python 依赖。

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
```

下载缓存好的 FineWeb 数据。这里先使用 1024-token 词表。

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

默认下载完整验证集 + 80 个训练分片（8B tokens）。如果只想小规模迭代，可传入 `--train-shards N`，例如 `--train-shards 1`。

启动你的第一轮训练。这里使用单卡 H100，因此 `nproc_per_node=1`。

```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

默认情况下，`train_gpt.py` 维持约 10 分钟 wallclock 上限。若希望更长训练，可显式覆盖，例如 `MAX_WALLCLOCK_SECONDS=0`。

默认该命令会在训练中输出 `train_loss`，结束时输出 `val_loss`、`val_bpb` 以及压缩后模型大小（在最终 `final_int8_zlib_roundtrip` 日志行）。如果你希望训练中周期性验证，可设置 `VAL_LOSS_EVERY`（如 `VAL_LOSS_EVERY=200`）。在 baseline 配置下，最终 `val_bpb` 通常约 ~1.2，压缩模型大小低于 16MB。

关于数据集导出、tokenizer 导出和 docs 缓存重建，请查看 [data/README.md](data/README.md)。

评测环境将使用 RunPod 镜像并已安装所有依赖。`requirements.txt` 仅作为自行搭建环境时的参考。

## 常见问题（FAQ）

**16MB 工件大小具体包含什么？**

提交工件大小 = 代码字节数 + 压缩模型字节数。所有计入大小的代码应放在 `train_gpt.py` 中。
上限是十进制 16MB，即 16,000,000 字节，而不是 16 MiB（16,777,216 字节）。
评测期间不允许外部下载、访问训练数据集或进行网络请求。工件必须完全自包含且可复现。

**OpenAI 会独立验证分数吗？**

我们不会自动验证每一个提交，但会逐步验证排行榜前列结果。任何不可复现结果都可能被取消资格。若你发现纪录存在问题或不可复现，请在对应 PR 提出，并提交 GitHub Issue 说明发现。

**什么算“外部算力”？例如我离线调超参是否公平？**

这个边界很难绝对清晰。当前我们保留对“不符合挑战精神”的结果取消资格的权利。比如做一些 Adam 超参搜索是可以的，但若有证据显示你通过不公平方式引入额外算力（例如暴力穷举夸张随机种子），则不会被允许。请自行把握，如有疑问可提问，不会有处罚。

**评测限制是什么？**

我们不接受在 8xH100 上评测耗时超过 10 分钟的提交（注意：这是在 10 分钟训练限制之外额外增加的限制）。除此之外，评测方式可以自由发挥。与 modded-nanogpt 一样，允许任意序列长度评测。显然，评测期间不得访问任何训练数据，除非你把这些数据的位数计入 <16MB 限制。我们鼓励你像优化训练方法一样激进地探索评测方法。

**新提交的接收流程是什么？**

由于提交全部公开，新的 SOTA 记录按 PR 创建时间顺序接收。排行榜更新可能因验证和审核需要时间，请在提交时考虑当前 SOTA PR。如下所述，想上榜的提交需以足够统计显著性超过当前 SOTA。否则，如果方案足够独特或有趣，也可能作为“非纪录提交”被接收。

## 提交流程

新的 SOTA 纪录需满足以下条件：

1. 必须至少比现有 SOTA 好 0.005 nats。与 modded-nanogpt 一样，由于运行间方差，需要提供足够日志证明达到 `p < 0.01` 且改进达到 0.005 nats。若只是系统优化带来速度提升且 ML 本身不变，可豁免该要求。

2. 若修改 tokenizer 或数据集，需明确证明 `val_bpb` 计算正确。修改 tokenizer 的提交会被更严格审查，因为 bug 可能造成不公平收益。

3. 必须可复现地在 8xH100 上 10 分钟内运行完成。

所有提交应以 Pull Request 形式提交，并且只在相应 `/records` 子目录新增一个文件夹，且包含以下文件。未满足完整要求的提交将不被接受。

1. `README.md`：以合理细节解释该提交。

2. `submission.json`：包含姓名、GitHub ID、`val_bpb` 和相关元数据（可参考示例运行）。

3. 训练日志（由脚本自动生成）。请展示具有统计显著性的胜出结果。通常提交 3 次训练的平均值即可。

4. `train_gpt.py` 脚本及其他依赖。注意：该脚本必须能在 records 文件夹内成功编译并运行。损坏脚本不会被接收。

### 非纪录提交

我们同样欢迎不一定超越 SOTA、但满足 16MB 工件上限且方案独特有趣的提交。强烈鼓励大家提交“奇思妙想”：包括非常规方法、尚未优化的中间结果，甚至有价值的负结果。我们很期待看到你的想法。我们仍会对非纪录提交保持较高标准，请在 README 中详细说明你的方案与结果。

我们还接受“无限算力轨道”的非纪录提交，即不以 10 分钟限制为目标的运行。请在 README 中明确标注。

非纪录提交流程与 SOTA 提交一致，如上所述。

#### 关于核心代码的 PR

`train_gpt.py` 与 `train_gpt_mlx.py` 旨在为新参与者提供良好的起点，而不是 SOTA 配置。我们会接受对这些脚本的调优、改进或简化 PR，但不应显著增加复杂度。最佳模型应放在 `/records` 目录中。

## 支持

加入 [OpenAI Discord server](https://discord.com/invite/openai)，前往 Parameter Golf 频道（#parameter-golf-discussions、#parameter-golf-announcements）提问交流。

本仓库改编自 `modded-nanogpt`，归属声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
