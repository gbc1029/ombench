# holos-synergy-repo

该仓库包含两个部分：

1) `experiment/`：与远端模型服务交互（REST `/task/solve`）
2) `memrl/`：MemRL 训练代码（已迁入）

`bridge/` 提供两者之间的联动适配层。

## 目录结构

```
holos-synergy-repo/
├── experiment/                # 远端模型交互
│   ├── client/                # HTTP 请求封装
│   ├── config/                # 默认请求参数
│   ├── prompt/                # Prompt 组装
│   └── scripts/               # CLI 示例
├── bridge/                    # 联动适配层
│   ├── adapters/              # LLM/Embedding 适配
│   ├── dataset/               # 数据集加载
│   └── runners/               # 联动训练入口
└── memrl/                     # MemRL 源码（原结构保留）
```

## 模型交互（experiment）

请求体格式（字段均可覆盖）：

```json
{
  "request_id": "omb-probe-v2",
  "benchmark": "onemillion",
  "task_id": "natural_science/9978/global",
  "model": "sii-holos/Qwen 3.5 397B A17B",
  "timeout": 240,
  "step_limit": 150,
  "system_prompt": "string",
  "user_prompt": "string"
}
```

### 安装

在项目根目录执行一次即可：

```bash
pip install -e .
```

### CLI 示例

```bash
python -m experiment.scripts.solve_task \
  --task-id natural_science/9978/global \
  --dry-run
```

- `--dry-run`：只打印请求体，不发送请求
- 不带 `--dry-run` 会实际调用服务

## MemRL 联动训练（bridge）

入口脚本：

```bash
python -m bridge.runners.run_onemillion_memrl \
  --dataset-dir /path/to/benchmark/cache/onemillion/dataset \
  --dry-run
```

默认数据目录：`datasets/OneMillion-Bench`（已迁入仓库）。如需使用其它位置，传 `--dataset-dir` 覆盖。

说明：
- `--dataset-dir` 指向 OneMillion-Bench 的数据目录
- `--dry-run` 禁止向远端模型发送请求（仅构造流程）
- 训练过程会将 `experiment` 的 `session_data.messages` 作为轨迹写入 MemRL 记忆
- 需要安装 `sentence-transformers`（本地嵌入使用）

## OMBench 评测（Synergy 接口）

评测模块位于 `ombench_eval/`，支持三种模式：

1) **rubric 模式**：输入 Synergy 生成的内容 + rubrics，进行评分
2) **plain 模式**：仅使用 dataset 的 `question` 生成回答（不注入 rubrics），再评分
3) **memrl 模式**：在 `plain` 基础上，加入 MemRL 训练产出的记忆上下文，再评分

### 评测提示词

评分提示词在 `ombench_eval/prompts.py`：
- `RUBRIC_JUDGE_SYSTEM_PROMPT`
- `build_rubric_judge_prompt(...)`

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

### 其它接口通道

评测接口抽象在 `ombench_eval/judge.py`：
- `SolveJudge`：走当前 `/task/solve` 接口
- `CallableJudge`：允许接入其它评测接口（外部函数/SDK）

### 评测输出

默认输出到 `outputs/results.jsonl`，也可用 `--output` 指定其它路径。

## 批量管线（生成 → 打分 → 训练）

批量管线将响应生成、rubric 打分和 MemRL 训练串联为一条自动化流水线，支持并发请求、失败重试和断点续跑。

入口脚本：`bridge/runners/run_batch_pipeline.py`

### 三阶段流程

1. **Generate**：从 dataset 批量构建 prompt 并发送请求，`ThreadPoolExecutor` 并发，每条完成立即收集
2. **Score**：从响应中提取 answer，结合 dataset 中的 rubrics 并发打分
3. **Train**：将响应和分数写入 MemRL 记忆（串行，`MemoryService` 自管并发）

### 基本用法

```bash
python -m bridge.runners.run_batch_pipeline \
  --mode direct --workers 4 --limit 100 --dry-run
```

### 两种生成模式

**direct 模式**：直接使用 `experiment/` 的 `SolveClient` 发送请求

```bash
python -m bridge.runners.run_batch_pipeline --mode direct --workers 4
```

**memrl 模式**：通过 `bridge/adapters/solve_llm.py` 的 `SolveLLM` 适配器发送请求

```bash
python -m bridge.runners.run_batch_pipeline --mode memrl --workers 4
```

### 跳过训练阶段

仅生成响应 + 打分，不写入 MemRL 记忆：

```bash
python -m bridge.runners.run_batch_pipeline --mode direct --no-train --workers 4
```

### 输出到文件

```bash
python -m bridge.runners.run_batch_pipeline --mode direct --output outputs/results.jsonl
```

### 断点续跑

中断后从上次结果继续（自动跳过已完成的 task_id）：

```bash
python -m bridge.runners.run_batch_pipeline --mode direct --output outputs/results.jsonl --resume outputs/results.jsonl
```

### 完整参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mode` | `direct` | 生成模式：`direct` / `memrl` |
| `--workers` | `4` | 并发数（generate 和 score 阶段） |
| `--limit` | `0` | 最大处理条数（0 = 全部） |
| `--max-retries` | `1` | 单条请求最大重试次数（指数退避） |
| `--no-train` | `false` | 跳过 MemRL 训练阶段 |
| `--dry-run` | `false` | 不发送实际请求 |
| `--output` | `outputs/results.jsonl` | 结果输出 JSONL 文件路径 |
| `--resume` | 无 | 从之前的 JSONL 结果断点续跑 |
| `--judge-model` | 同 `--model` | 打分模型（可与生成模型不同） |
| `--dataset-dir` | `datasets/OneMillion-Bench` | 数据集目录 |

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


- `experiment/client/solver.py`：HTTP 客户端封装（含重试）
- `experiment/config/settings.py`：默认请求参数 + 覆盖
- `bridge/adapters/solve_llm.py`：将 `/task/solve` 适配为 MemRL 的 `BaseLLM`
- `bridge/adapters/hash_embedder.py`：本地 hash embedder（避免远端 embedding 调用）
- `bridge/runners/run_onemillion_memrl.py`：串行联动训练入口
- `bridge/runners/batch_pipeline.py`：批量管线核心（`BatchPipeline` 类）
- `bridge/runners/run_batch_pipeline.py`：批量管线 CLI 入口

## 注意事项

- 目前服务不可交互时，请务必使用 `--dry-run`
- `memrl/` 目录保持原仓库结构不变，相关配置请按需修改
