# 数据工作流

本目录包含挑战所需的数据集下载辅助脚本与导出脚本。

标准本地目录结构：

- `data/datasets/<dataset_name>/`
- `data/tokenizers/`
- `data/manifest.json`
- `data/docs_selected.jsonl`
- `data/docs_selected.source_manifest.json`

## 下载已发布数据

可通过以下命令下载某个 tokenizer 变体对应的 FineWeb 缓存导出：

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

该命令会填充 `./data/datasets/fineweb10B_sp1024/` 与 `./data/tokenizers/`。
默认会下载完整验证集以及 8B 训练 tokens（80 个训练分片）。

如需下载更多训练分片，可传入 `--train-shards`：

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 180
```

下载器基于 manifest 驱动，可从更大的已发布导出中仅拉取训练分片前缀。以当前 `100_000_000` tokens 的分片大小计算，`10B` 重分词后的训练 tokens 对应 `100` 个训练分片：

```bash
MATCHED_FINEWEB_REPO_ID=your-hf-username/your-dataset-repo \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=your_50B_export_root \
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 100
```

验证集始终从固定的 `fineweb_val_*` 切分完整下载。仅使用前 `N` 个训练分片，意味着在同一份冻结且已打乱顺序的导出前缀上训练，因此数据顺序会与该 tokenizer 家族的 baseline 保持一致。

默认发布仓库为 `willdepueoai/parameter-golf`，导出内容位于该仓库子目录 `datasets/` 下。

## 基于已发布文档重建 Tokenizer

若你希望在完全相同的已选文档上重训 tokenizer，或重新导出分片，可对已发布 docs 缓存运行独立 retokenizer：

```bash
python3 data/download_hf_docs_and_tokenize.py \
	--repo-id your-hf-username/your-dataset-repo \
	--remote-root your_50B_export_root \
	--output-root /tmp/my_custom_tokenizer_export \
	--tokenizer-config ./data/tokenizer_specs.json
```

附带文件 `docs_selected.source_manifest.json` 包含 `docs_sha256`，可用于验证你重建时使用的文档列表与顺序与 baseline 导出完全一致。

## 常用参数

对于 CPU 开销较重的导出任务，以下参数通常有用：

```bash
MATCHED_FINEWEB_SP_BATCH_SIZE=2048
MATCHED_FINEWEB_TOKENIZER_THREADS=16
MATCHED_FINEWEB_TIKTOKEN_THREADS=16
MATCHED_FINEWEB_GPT2_DECODE_BATCH_SIZE=512
```

这些参数分别控制：分片导出时 tokenizer 批量编码大小、tokenizer 线程数、tiktoken 线程数，以及在 blobstore docs-cache 路径上的 GPT-2 批量解码大小。
