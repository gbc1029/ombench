---
license: apache-2.0
---

# $OneMillion-Bench

A bilingual (Global/Chinese) realistic expert-level benchmark for evaluating language agents across **5 professional domains**. The benchmark contains **400 entries** with detailed, weighted rubric-based grading criteria designed for fine-grained evaluation of domain expertise, analytical reasoning, and instruction following.

## Dataset Structure

Each subdirectory is a **Hugging Face subset** (configuration), and all data is in the **`test`** split.

```
$OneMillion-Bench/
├── economics_and_finance/
│   └── test.json      # 80 entries (40 EN + 40 CN, distinct questions)
├── healthcare_and_medicine/
│   └── test.json      # 80 entries (40 matched EN-CN pairs)
├── industry/
│   └── test.json      # 80 entries (40 matched EN-CN pairs)
├── law/
│   └── test.json      # 80 entries (40 EN + 40 CN, distinct questions)
├── natural_science/
│   └── test.json      # 80 entries (40 matched EN-CN pairs)
└── README.md
```

| Subset | Split | Entries |
|---|---|---|
| `economics_and_finance` | `test` | 80 |
| `healthcare_and_medicine` | `test` | 80 |
| `industry` | `test` | 80 |
| `law` | `test` | 80 |
| `natural_science` | `test` | 80 |

## Domains & Coverage

| Domain | Categories | Example Subcategories | Bilingual Mode |
|---|---|---|---|
| **Economics & Finance** | Investing, FinTech, Banking, Insurance, M&A | Equities, VC/PE, Cryptocurrency, Commodities | Separate questions per language |
| **Healthcare & Medicine** | Clinical Medicine, Basic Medicine, Pharma & Biotech | Hepatobiliary Surgery, Oncology, Nephrology, Dentistry | Matched translation pairs |
| **Industry** | Telecommunications, ML, Architecture, Semiconductors | Backend Dev, Chemical Engineering, Chip Design | Matched translation pairs |
| **Law** | Civil, Criminal, International, Corporate, IP, Labor | Contract Disputes, Criminal Defense, Copyright, M&A | Separate questions per language |
| **Natural Science** | Chemistry, Biology, Physics, Mathematics | Organic Chemistry, Condensed Matter, Molecular Biology | Matched translation pairs |

## Entry Schema

Each entry is a JSON object with 7 fields:

```jsonc
{
  "id": "uuid-string",            // globally unique identifier
  "case_id": 1,                   // links bilingual pairs (in matched-pair domains)
  "language": "en",               // "en" or "cn" (50/50 split in every file)
  "system_prompt": "",            // reserved (empty across all entries)
  "question": "...",              // expert-level evaluation prompt
  "tags": {
    "topics": [                   // 3-level taxonomy
      "Domain",                   //   e.g. "Economics and Finance"
      "Category",                 //   e.g. "Investing"
      "Subcategory"               //   e.g. "Equities"
    ],
    "time_sensitivity": {
      "time_sensitivity": "Time-agnostic",   // or "Weakly/Strongly time-sensitive"
      "year_month": "NA",                    // "YYYY-MM" when time-sensitive
      "day": "NA"                            // "DD" when applicable
    }
  },
  "rubrics": [                    // weighted grading criteria (11-37 per entry)
    {
      "rubric_number": 1,
      "rubric_detail": "...",     // specific grading criterion
      "rubric_weight": 5,         // positive = reward, negative = penalty
      "rubric_label": "..."       // category (see below)
    }
  ]
}
```

### Rubric Labels

| Label | Role | Typical Weight |
|---|---|---|
| Factual Information | Tests factual accuracy | +3 to +5 |
| Analytical Reasoning | Assesses depth of analysis | +3 to +5 |
| Structure and Formatting | Evaluates output organization | -2 to -4 (penalty) |
| Instructions Following | Checks compliance with task constraints | mixed |

## Quick Start

```python
import json

# Load a subset (test split)
with open("natural_science/test.json") as f:
    data = json.load(f)

# Filter English entries
en_entries = [e for e in data if e["language"] == "en"]

# Iterate with rubrics
for entry in en_entries[:1]:
    print(f"Topic: {' > '.join(entry['tags']['topics'])}")
    print(f"Question: {entry['question'][:200]}...")
    print(f"Rubrics ({len(entry['rubrics'])}):")
    for r in entry["rubrics"][:3]:
        print(f"  [{r['rubric_weight']:+d}] {r['rubric_label']}: {r['rubric_detail'][:80]}...")
```

Example output:

```
Topic: Natural Sciences > Chemistry > Organic Chemistry
Question: You are an expert in organic chemistry. A graduate student is researching ...
Rubrics (18):
  [+5] Factual Information: Correctly identifies the primary reaction mechanism ...
  [+4] Analytical Reasoning: Provides a coherent comparison of thermodynamic vs ...
  [-3] Structure and Formatting: Response lacks clear section headings or logica...
```

## Evaluation

Score a model response by summing the weights of satisfied rubrics:

```python
def score(response: str, rubrics: list, judge_fn) -> dict:
    """
    judge_fn(response, rubric_detail) -> bool
    """
    total, earned = 0, 0
    for r in rubrics:
        met = judge_fn(response, r["rubric_detail"])
        if met:
            earned += r["rubric_weight"]
        if r["rubric_weight"] > 0:
            total += r["rubric_weight"]
    return {"score": earned, "max_possible": total, "pct": earned / total if total else 0}
```

## License

Apache 2.0
