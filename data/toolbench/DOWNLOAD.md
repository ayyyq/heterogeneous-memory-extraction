# ToolBench Data Setup

ToolBench uses [StableToolBench](https://github.com/THUNLP-MT/StableToolBench) for stable, reproducible evaluation with cached API responses.

## Required Directory Structure

```
data/toolbench/
├── G1_instruction.json          # Solvable test instructions (163 queries)
├── G2_instruction.json          # Solvable test instructions (106 queries)
├── G3_instruction.json          # Solvable test instructions (61 queries)
├── test_query_ids/              # Solvable query ID lists
├── api_cache/
│   └── tool_response_cache/     # StableToolBench cached API responses (~522MB)
└── tools/                       # Tool/API documentation (~2.1GB)
```

**Only G1/G2/G3 instruction splits are used by the pipeline** (330 total solvable queries).

## Download Commands

All commands assume you are in the `MME_expr/` project root.

### Step 1: Clone StableToolBench (temporary, will be deleted at the end)

```bash
git clone https://github.com/THUNLP-MT/StableToolBench.git
```

### Step 2: Download API Cache + Tool Definitions

Download `server_cache.zip` from one of:
- **Tsinghua Cloud**: https://cloud.tsinghua.edu.cn/f/07ee752ad20b43ed9b0d/?dl=1
- **HuggingFace**: https://huggingface.co/datasets/stabletoolbench/Cache

```bash
wget -O StableToolBench/server/server_cache.zip \
  "https://cloud.tsinghua.edu.cn/f/07ee752ad20b43ed9b0d/?dl=1"

cd StableToolBench/server
unzip server_cache.zip
cd ../..
```

After unzipping, `StableToolBench/server/` should contain `tool_response_cache/` (~522MB) and `tools/` (~2.1GB).

### Step 3: Copy Solvable Test Instructions

```bash
mkdir -p data/toolbench/test_query_ids

cp StableToolBench/solvable_queries/test_instruction/G1_instruction.json data/toolbench/
cp StableToolBench/solvable_queries/test_instruction/G2_instruction.json data/toolbench/
cp StableToolBench/solvable_queries/test_instruction/G3_instruction.json data/toolbench/

# Optional splits (not used by pipeline)
cp StableToolBench/solvable_queries/test_instruction/G1_category.json data/toolbench/
cp StableToolBench/solvable_queries/test_instruction/G1_tool.json data/toolbench/
cp StableToolBench/solvable_queries/test_instruction/G2_category.json data/toolbench/

cp StableToolBench/solvable_queries/test_query_ids/*.json data/toolbench/test_query_ids/
```

### Step 4: Move Cache and Tools

```bash
mkdir -p data/toolbench/api_cache

mv StableToolBench/server/tool_response_cache data/toolbench/api_cache/tool_response_cache
mv StableToolBench/server/tools data/toolbench/tools
```

### Step 5: Clean up StableToolBench

```bash
rm -rf StableToolBench/
```

## Verification

```bash
python -c "
import json
for split in ['G1_instruction', 'G2_instruction', 'G3_instruction']:
    data = json.load(open(f'data/toolbench/{split}.json'))
    print(f'{split}: {len(data)} queries')
"
```
