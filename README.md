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
  "model": "holos-qwen35-397b/Qwen3.5-397B-A17B",
  "timeout": 240,
  "step_limit": 150,
  "system_prompt": "string",
  "user_prompt": "string"
}
```

### CLI 示例

```bash
python experiment/scripts/solve_task.py \
  --task-id natural_science/9978/global \
  --dry-run
```

- `--dry-run`：只打印请求体，不发送请求
- 不带 `--dry-run` 会实际调用服务

## MemRL 联动训练（bridge）

入口脚本：

```bash
python bridge/runners/run_onemillion_memrl.py \
  --dataset-dir /path/to/benchmark/cache/onemillion/dataset \
  --dry-run
```

默认数据目录：`datasets/OneMillion-Bench`（已迁入仓库）。如需使用其它位置，传 `--dataset-dir` 覆盖。

说明：
- `--dataset-dir` 指向 OneMillion-Bench 的数据目录
- `--dry-run` 禁止向远端模型发送请求（仅构造流程）
- 训练过程会将 `experiment` 的 `session_data.messages` 作为轨迹写入 MemRL 记忆

## 端到端流程

1) 数据读取
- `bridge/dataset/onemillion_loader.py` 从 OneMillion-Bench 数据目录加载 `test.json`
- 生成 `task_id = {subset}/{case_id}/{language}`

2) Prompt 构造
- `experiment/prompt/onemillion_prompt.py` 根据数据条目组装 prompt
- 产出 `system_prompt` + `user_prompt`

3) 远端请求
- `experiment/client/solver.py` 发送 `/task/solve` 请求
- `SolveSettings` 支持覆盖 `model/timeout/step_limit/request_id` 等字段
- `--dry-run` 模式仅返回 `{status: "dry_run", request: payload}`

4) 结果解析
- 读取返回 JSON 中的 `status` 与 `result.answer`
- `session_data.messages` 作为 MemRL 训练轨迹（原样保留）

5) MemRL 记忆写入
- `bridge/runners/run_onemillion_memrl.py` 调用 `MemoryService.add_memory`
- `task_description` 使用 `user_prompt`
- `trajectory` 使用 `session_data.messages` 序列化后的 JSON

6) 后续训练扩展
- 如需接入 MemRL 原始 runner，可在 `bridge/` 内新增适配器
- 不修改 `memrl/` 源码，仅通过 `bridge/` 进行联动

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
python ombench_eval/run_eval.py --mode plain --dry-run
```

#### rubric 模式（使用已有答案）

```bash
python ombench_eval/run_eval.py \
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
python ombench_eval/run_eval.py --mode plain --dry-run
```

#### memrl 模式（注入记忆）

```bash
python ombench_eval/run_eval.py \
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

默认输出到控制台 JSON；可以用 `--output` 写入文件。


- `experiment/client/solver.py`：HTTP 客户端封装
- `experiment/config/settings.py`：默认请求参数 + 覆盖
- `bridge/adapters/solve_llm.py`：将 `/task/solve` 适配为 MemRL 的 `BaseLLM`
- `bridge/adapters/hash_embedder.py`：本地 hash embedder（避免远端 embedding 调用）
- `bridge/runners/run_onemillion_memrl.py`：联动训练入口

## 注意事项

- 目前服务不可交互时，请务必使用 `--dry-run`
- `memrl/` 目录保持原仓库结构不变，相关配置请按需修改
