# holos-synergy-repo

该仓库包含三个部分：

1. `experiment/`：与远端模型服务交互（REST `/task/solve`）
2. `memrl/`：MemRL 训练代码（已迁入）
3. `ombench_eval/`：OMBench 评测模块（基于 rubric 的自动评分）

`bridge/` 提供两者之间的联动适配层。

## 目录结构

```
holos-synergy-repo/
├── experiment/                # 远端模型交互
│   ├── client/                # HTTP 请求封装 (SolveClient)
│   ├── config/                # 默认请求参数 (SolveSettings)
│   ├── prompt/                # Prompt 组装
│   └── scripts/               # CLI 示例
├── bridge/                    # 联动适配层
│   ├── adapters/              # LLM/Embedding 适配
│   ├── dataset/               # 数据集加载
│   └── runners/               # 批量管线 & 联动训练入口
├── ombench_eval/              # OMBench 评测模块
│   ├── evaluator.py           # 评分核心逻辑（单条 & 批量）
│   ├── judge.py               # Judge 实现（OpenAI 兼容 API）
│   ├── prompts.py             # 评分提示词模板
│   ├── run_eval.py            # 评测 CLI 入口
├── memrl/                     # MemRL 源码（原结构保留）
└── datasets/                  # 数据集
    └── OneMillion-Bench/
```

## 环境准备

### 安装

```bash
pip install -e .
```

### 环境变量

评分模块使用外部 OpenAI 兼容 API，需设置 API Key：

```bash
export INF_API_KEY="your_api_key_here"
```

## 快速验证

可用 `run_eval` 的 `--dry-run` 做请求预览：

```bash
python -m ombench_eval.run_eval --mode plain --dry-run
```

---

## 模型交互（experiment）

生成阶段通过 `/task/solve` 接口与远端模型交互。

请求体格式（字段均可通过 CLI 或代码覆盖）：

```json
{
  "request_id": "omb-probe-v2",
  "benchmark": "onemillion",
  "task_id": "natural_science/9978/global",
  "model": "qwen",
  "timeout": 1200,
  "step_limit": 150,
  "include_task_prompt": false,
  "system_prompt": "string",
  "user_prompt": "string"
}
```

> **注意**：`include_task_prompt` 硬编码为 `false`，仅出现在生成模型（`/task/solve`）请求中。评分模型（`OpenAIJudge`，走 `/v1/chat/completions`）和训练用 LLM（`OpenAILLM`）不包含该字段。

### CLI 示例

```bash
python -m experiment.scripts.solve_task \
  --task-id natural_science/9978/global \
  --dry-run
```

#### solve_task 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--task-id` | 必填 | 任务 ID |
| `--base-url` | `http://10.245.198.39:8000` | 模型服务地址 |
| `--endpoint` | `/task/solve` | 请求端点 |
| `--model` | `qwen` | 生成模型（别名：`qwen`/`nex`） |
| `--timeout` | `1200` | 请求超时（秒） |
| `--step-limit` | `150` | 步数限制 |
| `--dry-run` | `false` | 仅打印请求体，不发送 |

---

## OMBench 评测（ombench_eval）

评测模块位于 `ombench_eval/`，支持三种模式：

1. **rubric 模式**：输入已有回答 + rubrics 进行评分（不生成）
2. **plain 模式**：使用 dataset 的 question 生成回答（不注入 rubrics / domain），再评分
3. **memrl 模式**：在 plain 基础上注入 MemRL 记忆上下文，再评分

### 架构说明

**生成侧**（待评测模型）通过 `SolveClient` 调用 `/task/solve` 接口，保持不变。

**评分侧**（Judge 模型）使用 `OpenAIJudge`，直接调用 OpenAI 兼容的 `/v1/chat/completions` 接口：
- 默认 base_url：`https://holos.openapi-qb.sii.edu.cn`
- 默认模型：`qwen3.5-397b-a17b`
- 认证：`Bearer $INF_API_KEY`
- Qwen3.5 thinking 模式兼容：自动从 `message.reasoning` 提取内容（`content` 为 null 时）

### 评测入口

```bash
python -m ombench_eval.run_eval --mode plain --dry-run
```

#### rubric 模式（使用已有答案）

```bash
python -m ombench_eval.run_eval \
  --mode rubric \
  --responses-file /path/to/responses.jsonl \
  --dry-run
```

`responses.jsonl` 每行示例：
```json
{"task_id": "natural_science/9978/en", "answer": "..."}
```

#### plain 模式（不注入 rubrics）

```bash
python -m ombench_eval.run_eval --mode plain --dry-run
```

#### memrl 模式（注入记忆）

```bash
python -m ombench_eval.run_eval \
  --mode memrl \
  --memory-context /path/to/memory.json \
  --dry-run
```

`memory.json` 示例：
```json
{
  "natural_science/9978/en": "memory content..."
}
```

也支持 JSONL：
```json
{"task_id": "natural_science/9978/en", "memory": "..."}
```

#### run_eval 完整参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mode` | `plain` | 评测模式：`rubric` / `plain` / `memrl` |
| `--dataset-dir` | `datasets/OneMillion-Bench` | 数据集目录 |
| `--responses-file` | 无 | rubric 模式所需的已有答案文件 |
| `--memory-context` | 无 | memrl 模式的记忆上下文文件（JSON/JSONL） |
| `--limit` | `0` | 处理任务数量（0 = 全部） |
| `--output` | `outputs/results.jsonl` | 结果输出路径 |
| `--base-url` | `http://10.245.198.39:8000` | 生成模型服务地址 |
| `--endpoint` | `/task/solve` | 生成请求端点 |
| `--model` | `qwen` | 生成模型（别名：`qwen`/`nex`） |
| `--judge-model` | `qwen3.5-397b-a17b` | 评分模型 |
| `--judge-base-url` | `https://holos.openapi-qb.sii.edu.cn` | 评分 API 地址 |
| `--timeout` | `1200` | 生成请求超时（秒） |
| `--judge-timeout` | `600` | 评分请求超时（秒） |
| `--step-limit` | `150` | 生成步数限制 |
| `--dry-run` | `false` | 不发送实际请求 |

---

## 批量管线（生成 → 打分 → 训练）

批量管线将响应生成、rubric 打分和 MemRL 训练串联为一条自动化流水线，支持并发生成、逐条评分、实时文件写入、失败重试和断点续跑。

入口脚本：`bridge/runners/run_batch_pipeline.py`

### 三阶段流程

1. **Generate**：从 dataset 批量构建 prompt 并发送到 `/task/solve`，`ThreadPoolExecutor` 并发。每生成一条结果**立即写入** `outputs/generated.jsonl`
2. **Score**：每条生成结果完成后立即提交评分（`/v1/chat/completions`），按 `score_workers` 并发。每获得一条评分结果**立即写入** `outputs/scored.jsonl`
3. **Train**：将响应和分数写入 MemRL 记忆（串行，`MemoryService` 自管并发）。每训练一条**立即写入** `outputs/trained.jsonl`

### 实时写入行为

批量管线支持**逐条实时写入**，中途中断不会丢失已完成结果：

- `generated.jsonl`：每完成一条生成，立即追加写入
- `scored.jsonl`：每完成一条评分，立即追加写入
- `trained.jsonl`：每完成一条训练，立即追加写入
- 新运行时覆盖已有文件；断点续跑（`--resume-*`）时自动跳过已完成的 task_id

### 进度展示

运行时在 stdout 显示各阶段的实时进度条：

- 单批模式：`Generate: 45/100 [02:30<...]` `Score: 40/100 [01:20<...]` `Train: 38/100 [00:45<...]`
- 分批模式（`--gen-score-batch`）：显示**当前批次进度** + **整体进度**
  ```
  Generate [batch 1/3]: 100/100  Overall Generate:  100/300
  Score    [batch 1/3]: 100/100  Overall Score:     100/300
  Train    [batch 1/3]:  95/100  Overall Train:      95/300
  ```

### 基本用法

```bash
# 默认：4 并发生成，4 并发评分，全部三阶段
python -m bridge.runners.run_batch_pipeline \
  --mode plain --workers 4 --dry-run
```

### 生成模式

- **plain**：只用 `question` 生成
- **memrl**：`question + memory` 生成（需 `--memory-context`）

```bash
python -m bridge.runners.run_batch_pipeline --mode plain --workers 4
python -m bridge.runners.run_batch_pipeline --mode memrl --workers 4
```

### 选择执行阶段

通过 `--pipeline` 控制执行哪些阶段（逗号分隔）：

```bash
# 仅生成 + 评分，跳过训练
python -m bridge.runners.run_batch_pipeline --pipeline gen,score --workers 4

# 仅评分（需已有 generated.jsonl）
python -m bridge.runners.run_batch_pipeline --pipeline score

# 仅训练（需已有 scored.jsonl）
python -m bridge.runners.run_batch_pipeline --pipeline train
```

### 断点续跑

中断后从上次结果继续（自动跳过已完成的 task_id）：

```bash
python -m bridge.runners.run_batch_pipeline \
  --mode plain \
  --resume-gen-score outputs/scored.jsonl \
  --resume-train outputs/trained.jsonl
```

### 分批训练（memrl）

在 memrl 模式下，使用 `--gen-score-batch` 控制每批的 task 数量，每批完成后立即训练：

```bash
python -m bridge.runners.run_batch_pipeline \
  --mode memrl --pipeline gen,score,train \
  --gen-score-batch 50 --workers 4
```

### run_batch_pipeline 完整参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mode` | `plain` | 生成模式：`plain` / `memrl` |
| `--pipeline` | `gen,score,train` | 执行阶段（逗号分隔）：`gen` / `score` / `train` |
| `--dataset-dir` | `datasets/OneMillion-Bench` | 数据集目录 |
| `--base-url` | `http://10.245.198.39:8000` | 生成模型服务地址 |
| `--endpoint` | `/task/solve` | 生成请求端点 |
| `--model` | `qwen` | 生成模型（别名：`qwen`/`nex`） |
| `--timeout` | `1200` | 生成请求超时（秒） |
| `--step-limit` | `150` | 生成步数限制 |
| `--workers` | `4` | 生成阶段并发数 |
| `--max-retries` | `1` | 单条生成请求最大重试次数（指数退避） |
| `--judge-model` | `qwen3.5-397b-a17b` | 评分模型 |
| `--judge-base-url` | `https://holos.openapi-qb.sii.edu.cn` | 评分 API 地址 |
| `--judge-timeout` | `600` | 评分请求超时（秒） |
| `--score-workers` | `4` | 评分阶段并发数 |
| `--limit` | `0` | 随机抽样任务数量（0 = 全部） |
| `--dry-run` | `false` | 不发送实际请求 |
| `--generated-output` | `outputs/generated.jsonl` | 生成结果输出 JSONL |
| `--scored-output` | `outputs/scored.jsonl` | 评分结果输出 JSONL |
| `--trained-output` | `outputs/trained.jsonl` | 训练记录输出 JSONL |
| `--resume-gen-score` | 无 | 跳过已有的 gen/score task_id（JSONL 路径） |
| `--resume-train` | 无 | 跳过已有的 train task_id（JSONL 路径） |
| `--memory-context` | 无 | memrl 模式的记忆上下文文件（JSON/JSONL） |
| `--train-base-url` | `https://holos.openapi-qb.sii.edu.cn` | 训练 LLM API 地址 |
| `--train-endpoint` | `/v1/chat/completions` | 训练 LLM 端点 |
| `--train-model` | `qwen3.5-397b-a17b` | 训练 LLM 模型 |
| `--train-api-key-env` | `INF_API_KEY` | 训练 API Key 环境变量名 |
| `--checkpoint-dir` | `checkpoints/batch_pipeline` | 记忆 checkpoint 保存目录 |
| `--checkpoint-every` | `100` | 每训练 N 条保存一次 checkpoint |
| `--gen-score-batch` | `100` | 分批大小（memrl 模式 gen+score+train 循环） |
| `--load-checkpoint` | 无 | 启动时加载指定 checkpoint 目录 |

### 汇总统计

运行结束后自动输出统计：

```
========================================
  Total: 100 | Completed: 95 | Failed: 5
  Avg Score: 7.2/10.0 (72.0%)
  By subset:
    natural_science: 38/40 completed, avg 7.5/10.0
    law:             25/28 completed, avg 6.8/10.0
    ...
========================================
```

---

## MemRL 联动训练（bridge）

串行联动训练入口：

```bash
python -m bridge.runners.run_onemillion_memrl \
  --dataset-dir /path/to/benchmark/cache/onemillion/dataset \
  --dry-run
```

#### run_onemillion_memrl 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset-dir` | `datasets/OneMillion-Bench` | 数据集目录 |
| `--base-url` | `http://10.245.198.39:8000` | 模型服务地址 |
| `--endpoint` | `/task/solve` | 请求端点 |
| `--model` | `qwen` | 生成模型（别名：`qwen`/`nex`） |
| `--timeout` | `1200` | 请求超时（秒） |
| `--step-limit` | `150` | 步数限制 |
| `--dry-run` | `false` | 不发送实际请求 |

训练过程会将 `session_data.messages` 作为轨迹写入 MemRL 记忆。需要安装 `sentence-transformers`（本地嵌入使用）。

---

## 评测接口说明

评测接口抽象在 `ombench_eval/judge.py`：

| 类 | 说明 |
|---|---|
| `OpenAIJudge` | 调用 OpenAI 兼容的 `/v1/chat/completions` 接口评分 |
| `CallableJudge` | 允许接入其它评测接口（传入自定义函数） |
| `SolveJudge` | `OpenAIJudge` 的别名（向后兼容） |

`OpenAIJudge` 默认配置（`JudgeSettings`）：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `base_url` | `https://holos.openapi-qb.sii.edu.cn` | API 地址 |
| `model` | `qwen3.5-397b-a17b` | 评分模型 |
| `api_key_env` | `INF_API_KEY` | 读取 API Key 的环境变量名 |
| `timeout` | `600` | 请求超时（秒） |
| `max_tokens` | `16384` | 最大生成 token 数（避免截断） |

---

## 关键文件索引

| 文件 | 说明 |
|---|---|
| `experiment/client/solver.py` | HTTP 客户端封装（含重试） |
| `experiment/config/settings.py` | 默认请求参数 + 覆盖（`include_task_prompt` 硬编码为 `false`） |
| `bridge/adapters/solve_llm.py` | 将 `/task/solve` 适配为 MemRL 的 `BaseLLM` |
| `bridge/adapters/hash_embedder.py` | 本地 hash embedder（避免远端 embedding 调用） |
| `bridge/runners/batch_pipeline.py` | 批量管线核心（`BatchPipeline` 类，实时写入 + 进度展示） |
| `bridge/runners/run_batch_pipeline.py` | 批量管线 CLI 入口 |
| `bridge/runners/run_onemillion_memrl.py` | 串行联动训练入口 |
| `ombench_eval/evaluator.py` | 评分核心（`score_response` + `score_responses_batch`） |
| `ombench_eval/judge.py` | Judge 实现（`OpenAIJudge`） |
| `ombench_eval/prompts.py` | 评分提示词（单条 + 批量模板） |
| `ombench_eval/run_eval.py` | 评测 CLI 入口 |

## 注意事项

- 评分需要设置 `export INF_API_KEY="your_key"`
- 目前服务不可交互时，请务必使用 `--dry-run`
- `memrl/` 目录保持原仓库结构不变，相关配置请按需修改
- Qwen3.5 thinking 模式下 `content` 可能为 null，评分模块会自动从 `reasoning` 字段提取内容
- 生成请求始终携带 `include_task_prompt: false`，该参数不传递给评分模型或训练 LLM
