from __future__ import annotations

from typing import Any, Dict, List, Optional

RUBRIC_JUDGE_SYSTEM_PROMPT = ""

_RUBRIC_TEMPLATE = """\
## 角色与核心任务

**角色：** 你是一名公正、精确且严格的AI响应评估裁判。

**核心任务：** 根据详细的评分标准（Rubric），对大型语言模型的回复（modelResponse）进行逐项评估。你需要判断模型回复是否符合评分标准中的具体描述。

**评估原则：**

1. **寻找直接证据：** 评估必须严格依据模型回复中**实际存在**的文本证据。不能进行主观猜测或过度解读。只有明确指出的内容才算数。

2. **二元判断（是/否）：** 每一个 Rubric 项的评估结果只有两种：
   - **命中 (是)**：模型回复中确实包含或命中了rubric描述的内容或特征。
   - **未命中 (否)**：模型回复中没有包含或没有命中rubric描述的内容或特征。

   *注意：这一逻辑通用于正分项（得分点）和负分项（扣分点）。只要rubric里的描述发生了，就是"命中/是"。*

3. **评分规则：**
   - **正向得分项（rubricWeight > 0）**：输出"是"代表得到该项分数，输出"否"代表不得分（0分）。
   - **负向扣分项（rubricWeight < 0）**：输出"是"代表需要扣分（扣除对应分值），输出"否"代表不扣分（0分）。

---

## 评分步骤

请保持冷静和专注，严格遵循以下步骤：

**步骤一：理解上下文**
仔细阅读用户问题（prompt）、模型回复（modelResponse）、评分标准（rubric）。

**步骤二：判断是否命中**
对照评分标准（rubric）的描述，检查模型回复：
- 如果回复中**出现**了rubric描述的情况（无论是好的行为还是坏的错误），状态为 **"命中"**，结论输出 **"是"**。
- 如果回复中**未出现**rubric描述的情况，状态为 **"未命中"**，结论输出 **"否"**。

**步骤三：自我反思与格式化**
- 检查证据是否充分支持你的"是/否"判断。
- 严格按照JSON格式输出。

---

## 输出格式

对每条Rubric，输出一个JSON对象，包含以下字段：

```json
[
  {{
    "rubric_id": 1,
    "status": "是",
    "justification": "模型回复完整说明了问题X，符合评分要求"
  }},
  {{
    "rubric_id": 2,
    "status": "否",
    "justification": "模型回复未提及关键点Y"
  }}
]
```

**字段说明**：
- `rubric_id`：Rubric编号
- `status`：**"是"** 或 **"否"**
- `justification`：简要的中文评估依据（1-2句话）

---

## 输入信息

### 用户问题（prompt）
{prompt}

---

### AI回复（modelResponse）
{model_response}

---

### 评分项（Rubrics）
{rubrics_str}

---

请逐条评估所有Rubric，输出完整JSON数组。"""


def _format_rubrics(rubrics: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for rubric in rubrics:
        rubric_num = rubric.get("rubric_number", 0)
        rubric_detail = rubric.get("rubric_detail", "")
        rubric_weight = rubric.get("rubric_weight", 0)
        rubric_label = rubric.get("rubric_tag", rubric.get("rubric_label", ""))
        part = f"**Rubric {rubric_num}**\nrubricDetail: {rubric_detail}\nrubricWeight: {rubric_weight:+d}分"
        if rubric_label:
            part += f"\nrubricLabel: {rubric_label}"
        parts.append(part)
    return "\n\n".join(parts)


def build_rubric_judge_prompt(
    *,
    question: str,
    response: str,
    rubrics: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
) -> str:
    prompt_text = question
    if system_prompt:
        prompt_text = f"{system_prompt}\n\n{question}"

    rubrics_str = _format_rubrics(rubrics)
    return _RUBRIC_TEMPLATE.format(
        prompt=prompt_text,
        model_response=response,
        rubrics_str=rubrics_str,
    )
