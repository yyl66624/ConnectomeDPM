# ConnectomeDPM 开发文档

> 项目状态：开发准备阶段  
> 项目定位：基于真实连接组拓扑先验的选择性参数修正系统  
> 开发原则：新仓库、独立实现、先完成 H1 主线，再扩展 H2/H3  
> 默认训练环境：Linux + 单张 NVIDIA RTX 5090 级 GPU

---

## 1. 开发目标

ConnectomeDPM 的第一阶段目标不是复现 BitDPM 的完整工程，而是实现一个最小、可验证、可扩展的研究系统：

1. 冻结一个基础语言模型；
2. 构建固定的参数修正块库；
3. 基于 MaleCNS 构建一个有向图先验；
4. 将请求特征映射到图节点；
5. 通过固定图传播产生结构化路由表征；
6. 预测每个参数块的 `fix / break / unchanged` 概率；
7. 在独立校准集上学习是否干预；
8. 在固定覆盖率和部署阈值两种协议下评估；
9. 与严格匹配的 null topology 对照；
10. 在 H1 成立后，再进入 H2 拓扑机制分解和 H3 人工拓扑蒸馏。

首版不做：

- 多块同时激活；
- 动态 LoRA scale；
- 低比特量化；
- CPU/GPU 混合调度；
- Male/Female residual gating；
- 图扰动置信度作为核心方法；
- 端侧推理优化；
- 复杂 MoE；
- 任何脑区到语言能力的人工语义映射。

---

## 2. 最终研究主线

### H1 — Transfer

验证真实 MaleCNS 拓扑是否能够在严格匹配的 null model 下，提高：

- Net Repair；
- Break Risk；
- OOD 泛化；
- Router Sample Efficiency。

核心比较：

```text
MLP
No-Neighbor
Random Graph
Degree-Preserving Graph
Module-Preserving Graph
MaleCNS Graph
```

### H2 — Mechanism

仅当 H1 成立后执行。

定位收益究竟来自哪些结构因素：

```text
Directionality
Edge Weight
Degree Structure
Community Structure
Rich-Club / Hub Organization
Cross-Module Bottleneck
Motif Structure
```

Male/Female connectome 在此阶段只作为 biological replication，不作为“不同情境专家”。

### H3 — Abstraction

仅当 H2 找到稳定结构规律后执行。

从 MaleCNS 中提取有效组织原则，构建 synthetic topology，并测试：

```text
Synthetic Distilled Graph
vs.
MaleCNS Graph
vs.
Matched Null Graphs
```

目标是判断能否把真实 connectome 中的结构规律抽象为通用人工路由拓扑。

---

## 3. 仓库建议结构

```text
ConnectomeDPM/
├── README.md
├── DEVELOPMENT.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── model/
│   │   ├── qwen25_05b.yaml
│   │   └── qwen25_15b.yaml
│   ├── graph/
│   │   ├── malecns_k32.yaml
│   │   ├── malecns_k64.yaml
│   │   └── null_graphs.yaml
│   ├── router/
│   │   └── transfer_operator.yaml
│   ├── block/
│   │   └── block_bank_8.yaml
│   └── experiment/
│       ├── h1_main.yaml
│       ├── h1_sample_efficiency.yaml
│       ├── h2_mechanism.yaml
│       └── h3_distillation.yaml
│
├── connectomedpm/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── splits.py
│   │   ├── dedup.py
│   │   ├── task_registry.py
│   │   └── schemas.py
│   │
│   ├── backbone/
│   │   ├── loader.py
│   │   ├── feature_extractor.py
│   │   └── generation.py
│   │
│   ├── blocks/
│   │   ├── parameter_block.py
│   │   ├── block_bank.py
│   │   ├── inject.py
│   │   ├── trainer.py
│   │   └── manifest.py
│   │
│   ├── graph_prior/
│   │   ├── malecns_loader.py
│   │   ├── filter.py
│   │   ├── coarse_grain.py
│   │   ├── normalize.py
│   │   ├── features.py
│   │   ├── null_models.py
│   │   ├── graph_stats.py
│   │   └── manifest.py
│   │
│   ├── router/
│   │   ├── input_projector.py
│   │   ├── transfer_operator.py
│   │   ├── action_head.py
│   │   ├── loss.py
│   │   ├── train.py
│   │   └── inference.py
│   │
│   ├── supervision/
│   │   ├── candidate_cache.py
│   │   ├── outcome_labeler.py
│   │   ├── scorer.py
│   │   └── schemas.py
│   │
│   ├── calibration/
│   │   ├── threshold.py
│   │   ├── risk.py
│   │   └── coverage.py
│   │
│   ├── evaluation/
│   │   ├── fixed_coverage.py
│   │   ├── deployment.py
│   │   ├── sample_efficiency.py
│   │   ├── ood.py
│   │   ├── bootstrap.py
│   │   └── metrics.py
│   │
│   ├── audit/
│   │   ├── routing_trace.py
│   │   ├── leakage.py
│   │   ├── graph_usage.py
│   │   └── reproducibility.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── hash.py
│       ├── io.py
│       ├── logging.py
│       └── device.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── build_malecns_graph.py
│   ├── build_null_graphs.py
│   ├── train_blocks.py
│   ├── generate_candidate_cache.py
│   ├── train_router.py
│   ├── calibrate_router.py
│   ├── evaluate_h1.py
│   ├── evaluate_sample_efficiency.py
│   ├── evaluate_h2.py
│   └── evaluate_h3.py
│
├── experiments/
│   ├── manifests/
│   ├── caches/
│   ├── checkpoints/
│   ├── traces/
│   ├── results/
│   └── reports/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── manifests/
│
├── tests/
│   ├── test_graph_build.py
│   ├── test_null_models.py
│   ├── test_block_injection.py
│   ├── test_candidate_cache.py
│   ├── test_transfer_operator.py
│   ├── test_outcome_labels.py
│   ├── test_calibration.py
│   └── test_metrics.py
│
└── notebooks/
    └── diagnostics/
```

---

## 4. 数据分区

必须在任何正式实验前冻结：

```text
D_block
D_router_train
D_router_dev
D_cal
D_test_ID
D_test_OOD
```

### D_block

用途：

- 训练 8 个参数块；
- 块内验证；
- 固定 rank；
- 固定 scale；
- 固定 block manifest。

禁止：

- 根据 D_test 结果重新选择块；
- 根据 router 结果重新训练块。

### D_router_train

用途：

- 训练 Router；
- 使用 candidate outcome labels；
- 学习 `fix / break / unchanged`。

### D_router_dev

用途：

- Router 模型选择；
- hidden dimension；
- training epoch；
- learning rate；
- architecture 选择。

### D_cal

用途：

- 选择部署 threshold；
- 估计 break risk；
- 校准 coverage。

禁止：

- 修改图；
- 修改 block；
- 修改 Router architecture。

### D_test_ID

用途：

- 最终同分布测试。

### D_test_OOD

用途：

- 未见任务族；
- 未见模板族；
- 测试 inductive bias 是否可迁移。

### 分割原则

必须按以下粒度去重和分割：

```text
source
task family
template family
near-duplicate group
```

同一模板生成的改写不能跨 train / test。

---

## 5. 基础模型

首轮：

```text
Qwen2.5-0.5B-Instruct
```

用途：

- pipeline 验证；
- 数据检查；
- graph operator 检查；
- candidate cache 检查；
- 路由训练稳定性；
- null graph pipeline。

主实验：

```text
Qwen2.5-1.5B-Instruct
```

后续规模验证可使用更大模型，但必须在 H1 主结果稳定以后进行。

### Backbone 原则

- Backbone 冻结；
- BF16 优先；
- 不在 H1 中加入量化；
- generation 参数全实验固定；
- tokenizer 和模型版本写入 manifest；
- prompt template 固定；
- decoding config 固定。

---

## 6. Parameter Block 设计

首版固定：

```text
8 blocks
1 block maximum per request
+ base fallback action
```

每个 block 应对应一个可验证的错误族，而不是人为能力标签。

建议候选类型：

```text
numerical arithmetic
unit conversion
format following
code boundary condition
structured extraction
logic constraint
short factual correction
symbolic manipulation
```

最终 block 类型由数据决定。

### Block 必须记录

```yaml
block_id:
base_model:
base_model_revision:
target_layers:
target_modules:
rank:
scale:
train_split_hash:
validation_split_hash:
training_seed:
checkpoint_sha256:
```

### Block Gate

在进入 Graph Router 之前必须先检查：

```text
Oracle(block bank) > Base
```

如果 Oracle headroom 不足，暂停 Router 开发。

最少报告：

```text
Base Accuracy
Oracle Accuracy
Oracle Fixes
Oracle Breaks
Oracle Net Repair
Per-Block Fix Count
Per-Block Break Count
```

---

## 7. Candidate Outcome Cache

这是整个项目的核心缓存。

对于每个样本：

```text
Base
Block_1
Block_2
...
Block_8
```

全部离线执行一次。

保存：

```json
{
  "sample_id": "...",
  "split": "...",
  "base_correct": true,
  "actions": {
    "base": {
      "correct": true,
      "score": 1
    },
    "block_01": {
      "correct": true,
      "outcome": "unchanged"
    },
    "block_02": {
      "correct": false,
      "outcome": "break"
    }
  }
}
```

Outcome 定义：

```text
Base wrong -> Block correct = FIX
Base correct -> Block wrong = BREAK
Otherwise = UNCHANGED
```

所有 Router 必须共享同一份 candidate cache。

### Cache Key

必须绑定：

```text
base model
model revision
block manifest
prompt template
decoding config
scorer
dataset split
```

任一变化都应产生新的 cache hash。

---

## 8. MaleCNS Graph Pipeline

主版本：

```text
MaleCNS v1.0
Central Brain
K = 32 routing nodes
```

敏感性：

```text
K = 64
```

### Step 1 — Raw Data

保留：

```text
connectome edge weights
body annotations
community assignment
mapping metadata
version manifest
```

必须记录：

```text
download source
download date
file size
SHA256
dataset version
```

### Step 2 — Main Node Set

主节点集合：

```text
Central-Brain community neuron IDs
∩
MaleCNS v1.0 valid annotation IDs
```

连接只保留：

```text
src ∈ main_node_set
dst ∈ main_node_set
```

### Step 3 — Community Aggregation

先聚合原始社区之间的有向边权。

禁止：

- 看语言任务；
- 看 block performance；
- 看 Router 结果；
- 看 D_test。

### Step 4 — Deterministic Coarse-Graining

将原始社区确定性地压缩到：

```text
K = 32
```

需要保存完整 mapping：

```text
neuron
-> original community
-> coarse routing node
```

### Step 5 — Edge Transform

主实验：

```python
w = log1p(raw_synapse_count)
```

分别构建：

```text
incoming transition operator
outgoing transition operator
```

对角线：

```text
A[i, i] = 0
```

原 self-loop 单独保存为 node feature。

### Step 6 — Graph Artifact

必须输出：

```text
nodes.parquet
edges.parquet
adjacency.npz
transition_in.npz
transition_out.npz
node_features.npy
coarse_mapping.json
graph_stats.json
manifest.json
```

---

## 9. Graph-Forced Transfer Operator

首版不采用高自由度 GNN。

目标是确保 Router 不能轻易绕过 topology。

### Input Projection

基础模型得到 request feature：

\[
h_x \in \mathbb{R}^{d}
\]

映射到 K 个图节点：

\[
s(x)=\operatorname{softmax}(W_{in}h_x)
\]

其中：

```text
K = 32
```

### Fixed Graph Propagation

对真实图：

\[
z_{out}^{(1)} = P_{out}s
\]

\[
z_{out}^{(2)} = P_{out}^{2}s
\]

\[
z_{in}^{(1)} = P_{in}s
\]

\[
z_{in}^{(2)} = P_{in}^{2}s
\]

组合：

\[
z(x)=
[s,
P_{in}s,
P_{out}s,
P_{in}^{2}s,
P_{out}^{2}s]
\]

首版不要加入任意深的 learnable message passing。

### Action Head

对每个 block 输出：

```text
P(fix)
P(break)
P(unchanged)
```

例如：

\[
p_b=W_bz(x)
\]

最终 action score：

\[
S_b
=
P(\mathrm{fix})
-\lambda P(\mathrm{break})
-\gamma C_b
\]

首版可先令：

```text
gamma = 0
```

先只研究质量。

### Fallback

如果：

```text
max_b S_b < threshold
```

则：

```text
BASE
```

否则：

```text
argmax_b S_b
```

最多激活一个 block。

---

## 10. Router Loss

每个 action 是三分类：

```text
FIX
BREAK
UNCHANGED
```

基础损失：

```text
Cross Entropy
```

由于 BREAK 通常少但代价高，可使用 class weight：

```text
w_break > w_fix >= w_unchanged
```

权重只能在 `D_router_dev` 上确定。

不要根据 `D_test` 调节。

需要同时记录：

```text
action-level CE
fix AUROC
break AUROC
fix PR-AUC
break PR-AUC
top-1 action accuracy
net utility
```

---

## 11. Null Topology

### N0 — No Neighbor

关闭全部邻接传播：

```text
z = s
```

用于检查：

```text
graph edges 是否真的有贡献
```

### N1 — Uniform Random

仅作弱基线。

不能作为主要生物特异性对照。

### N2 — Degree-Preserving

在粗化后的有向支持图上重连：

```text
preserve in-degree
preserve out-degree
preserve node count
preserve edge count
```

边权重新置换，尽量保持全局 weight distribution。

### N3 — Module-Preserving

至少匹配：

```text
module sizes
within-module density
between-module density
block-to-block edge count
```

若不能完全同时匹配，应报告实际偏差。

### N4+ — H2 Null Ladder

仅在 H1 后使用。

逐步恢复：

```text
degree
weight
community
hub/rich-club
direction
motif
```

用于结构机制定位。

---

## 12. 所有图必须公平比较

真实图和所有 null graph 必须使用完全相同：

```text
input projector
action head
hidden dimension
optimizer
learning rate search space
training steps
early stopping rule
router train split
router dev split
calibration protocol
block bank
candidate cache
```

每个 graph 必须重新计算自己的：

```text
degree
strength
node features
transition operator
graph statistics
```

禁止把 MaleCNS node features 直接复制到随机图。

---

## 13. H1 主实验

### 13.1 Fixed-Coverage Evaluation

主 coverage：

```text
10%
```

附加：

```text
5%
20%
```

流程：

1. 每个方法只根据自己的 Router score 排序；
2. 选择相同比例请求；
3. 每个请求只执行其 top-1 block；
4. 统一使用 candidate cache；
5. test answer 不能用于排序。

主要指标：

```text
Fixes
Breaks
Net Repair
Net Repair per 1000
Selective Break Risk
Total Accuracy Delta
```

### 13.2 Deployment Evaluation

使用 `D_cal`：

```text
learn threshold
```

冻结后测试：

```text
actual coverage
fixes
breaks
break risk
net repair
```

固定 coverage 和 deployment threshold 的结果必须分开报告。

---

## 14. Sample Efficiency

Router supervision size：

```text
1%
5%
10%
25%
50%
100%
```

每个比例必须：

- 保持 task-family composition；
- 使用固定 sample IDs；
- 多个 seed 重复；
- 所有方法使用完全相同的训练子集。

比较：

```text
MLP
No-Neighbor
Degree-Preserving
Module-Preserving
MaleCNS
```

主图：

```text
x-axis: Router Supervision Fraction
y-axis: Net Repair @ 10% Coverage
```

同时可画：

```text
Break Risk
OOD Net Repair
```

如果 MaleCNS 主要在低数据条件下获益，这将被解释为 inductive-bias evidence，而不是容量优势。

---

## 15. OOD Evaluation

OOD 必须是：

```text
unseen task family
```

而不是简单随机 test split。

必须在实验前冻结。

报告：

```text
ID Net Repair
OOD Net Repair
ID Break Risk
OOD Break Risk
Coverage Shift
Action Distribution Shift
```

---

## 16. H2 — Topology Mechanism

仅当 H1 满足进入条件后执行。

### 机制消融

分别测试：

```text
Direction Shuffle
Weight Shuffle
Degree-Preserving Rewire
Community-Preserving Rewire
Hub/Rich-Club Lesion
Cross-Module Bottleneck Lesion
Motif Perturbation
```

每次只改变一个主要结构因素。

### Biological Replication

Male/Female connectome 用于：

```text
Male graph
Female graph
Shared structural pattern
Matched null graphs
```

目标：

```text
收益是否跨真实 connectome 复现
```

不赋予语言任务性别意义。

---

## 17. H3 — Distilled Synthetic Topology

仅在 H2 找到稳定结构规律后执行。

根据 H2 结果构建 synthetic graph。

可能包含：

```text
modular organization
hub/rich-club core
directed asymmetric flow
sparse cross-module bottlenecks
```

要求：

```text
K = 32
same parameter budget
same router
same training data
same candidate cache
```

比较：

```text
MaleCNS
Distilled Synthetic
Degree Null
Module Null
MLP
```

如果 synthetic graph 可稳定复制 MaleCNS 收益，则说明发现的是 transferable wiring principle，而不是某个具体生物图实例的偶然结构。

---

## 18. 核心指标

### Fix Rate

\[
\frac{\#Fix}{\#BaseWrong}
\]

### Overall Break Rate

\[
\frac{\#Break}{\#BaseCorrect}
\]

### Selective Break Risk

\[
\frac{\#Break}
{\#Intervened \cap BaseCorrect}
\]

### Net Repair

\[
\#Fix-\#Break
\]

### Net Repair Rate

\[
\frac{\#Fix-\#Break}{N}
\]

### Net Repair per 1000

\[
1000\cdot
\frac{\#Fix-\#Break}{N}
\]

### Coverage

\[
\frac{\#Intervened}{N}
\]

所有指标必须同时保存：

```text
numerator
denominator
confidence interval
seed
graph instance
```

---

## 19. 随机性与统计

### Pilot

```text
3 training seeds
```

### Main Confirmation

MaleCNS：

```text
5 training seeds
```

每类随机图：

```text
5 graph instances
×
5 training seeds
```

### Formal Topology Randomization

若 H1 成立并需要正式 topology randomization test：

```text
>= 99 null graph instances
```

### Bootstrap

以：

```text
task family
or
template group
```

作为 cluster 做 paired bootstrap。

不要把同一模板生成的 prompt 当作完全独立样本。

---

## 20. 实验编号

建议固定编号，避免后期混乱。

```text
R00  Environment + Manifest
R01  Data Split + Dedup
R02  MaleCNS Graph Build
R03  Null Graph Build
R04  Block Bank Training
R05  Oracle Headroom Check
R06  Candidate Cache Generation
R07  Router Pipeline Sanity
R08  H1 Pilot
R09  H1 Main
R10  Sample Efficiency
R11  OOD Confirmation
R12  Calibration / Deployment
R13  H2 Mechanism
R14  Biological Replication
R15  H3 Synthetic Distillation
R16  Final Statistical Confirmation
```

每个 run 必须生成独立 manifest。

---

## 21. Run Manifest

每次实验至少保存：

```yaml
run_id:
git_commit:
timestamp:

model:
  name:
  revision:
  dtype:

dataset:
  manifest_hash:
  split_hash:

block_bank:
  manifest_hash:

graph:
  graph_type:
  graph_id:
  manifest_hash:

router:
  config_hash:

candidate_cache:
  cache_hash:

seed:
training_seed:
graph_seed:
sampling_seed:

hardware:
  gpu:
  cuda:
  torch:
```

---

## 22. 输出目录规范

```text
experiments/
└── results/
    └── R09_H1_MAIN/
        ├── manifest.yaml
        ├── metrics.json
        ├── predictions.parquet
        ├── routing_trace.parquet
        ├── bootstrap.json
        ├── graph_stats.json
        └── logs.txt
```

禁止只保存最终平均值。

必须保留 sample-level decision。

---

## 23. Routing Trace

每个请求至少记录：

```text
sample_id
split
base_correct
request_feature_hash
graph_id
router_score_per_block
predicted_fix_per_block
predicted_break_per_block
selected_action
fallback_or_intervene
selected_block
actual_outcome
coverage_rank
threshold
latency
```

这是后续：

```text
error analysis
mechanism analysis
reproducibility
paper case study
```

的基础。

---

## 24. Leakage Audit

正式实验前必须自动检查：

```text
train/test duplicate
template overlap
candidate label leakage
test answer leakage
graph construction leakage
block-selection leakage
threshold leakage
```

尤其禁止：

```text
D_test -> block selection
D_test -> graph construction
D_test -> router hyperparameter
D_test -> threshold
```

---

## 25. Graph Usage Audit

Router 训练完成后必须检查：

### A. No-Neighbor

```text
disable P_in / P_out
```

### B. Edge Rewire

保持其他模型参数不变，替换 graph。

### C. Node Permutation

同步重排：

```text
adjacency
node features
projection/readout correspondence
```

### D. Graph Sensitivity

记录：

```text
action-change rate
score correlation
net-repair change
```

如果 graph 改变后路由几乎不变，则不能声称 topology 起作用。

---

## 26. 服务器开发顺序

### Phase A — 工程闭环

完成：

```text
R00-R07
```

要求：

- graph 可构建；
- block 可注入；
- cache 可生成；
- label 正确；
- router 可训练；
- inference 可运行；
- manifest 完整；
- graph usage sanity 通过。

### Phase B — H1 Pilot

执行：

```text
R08
```

只允许：

```text
0.5B
3 seeds
少量 null graph
```

目的：

```text
找 bug
检查方向
估计方差
```

禁止将 pilot 当最终结论。

### Phase C — H1 Main

执行：

```text
R09-R12
```

主模型：

```text
Qwen2.5-1.5B
```

完成：

```text
Main comparison
Sample efficiency
OOD
Calibration
```

### Phase D — H2

只有 H1 通过后执行：

```text
R13-R14
```

### Phase E — H3

只有 H2 得到可解释结构规律后执行：

```text
R15
```

### Phase F — Final Confirmation

执行：

```text
R16
```

冻结：

```text
code
data
graph
block bank
router config
threshold
statistics protocol
```

然后运行最终确认结果。

---

## 27. Gate 条件

### Gate 0 — Block Bank

进入 Router 前：

```text
Oracle Net Repair > 0
```

并且至少多个 block 有真实 fix 能力。

否则：

```text
STOP
```

### Gate 1 — Graph Usage

必须：

```text
Real Graph != No-Neighbor
```

至少在动作或 score 上表现出稳定差异。

否则：

```text
STOP topology claim
```

### Gate 2 — H1

H1 需要至少满足：

```text
MaleCNS > matched null
```

并且：

```text
差异不只来自 coverage
差异不只来自更高 break tolerance
差异不只来自单一 seed
差异不只来自单一 task family
```

### Gate 3 — H2

必须找到一个可重复结构因素：

```text
lesion -> performance drop
or
restoration -> performance recovery
```

### Gate 4 — H3

Distilled topology 必须至少：

```text
稳定超过 matched null
```

最好能接近 MaleCNS。

---

## 28. 失败处理

### Oracle 不够强

原因优先级：

```text
Block training
Block diversity
Target layer
Scale
Task design
Scorer
```

不要先改 Router。

### Router 学不会

检查：

```text
label imbalance
feature quality
action ambiguity
train/dev leakage
projection bottleneck
loss weights
```

### Real Graph ≈ No-Neighbor

结论：

```text
graph edges 未提供有效信息
```

先检查实现。

实现无误后停止 connectome-topology 主张。

### Real Graph > MLP，但 ≈ Module Null

结论：

```text
一般模块化结构有用
但 MaleCNS biological specificity 未获支持
```

### Real Graph > Degree Null，但 ≈ Module Null

重点转向：

```text
community structure
```

### Real Graph > All Matched Null

进入 H2。

### H2 找不到单一机制

可报告：

```text
distributed structural effect
```

但不能人为挑一个最好看的统计量做机制结论。

### H3 Synthetic ≈ MaleCNS

这是理想结果：

```text
发现可抽象 wiring principle
```

### H3 Synthetic < MaleCNS，但 > Null

结论：

```text
部分结构规律可迁移
仍存在未解释的 topology information
```

---

## 29. 测试要求

每次 push 前至少运行：

```bash
pytest tests/ -q
```

核心单元测试：

```text
graph node count
edge direction
coarse-grain determinism
null graph constraints
transition normalization
block injection
block isolation
candidate label correctness
cache hash
router shape
fallback behavior
metric denominator
split isolation
```

---

## 30. 第一批开发任务

### P0

- [ ] 初始化新 GitHub 仓库
- [ ] 建立目录结构
- [ ] 建立 Python 环境
- [ ] 建立 config loader
- [ ] 建立 manifest 系统
- [ ] 建立 seed 系统
- [ ] 建立 logging
- [ ] 建立 hash 工具

### P1

- [ ] 数据 schema
- [ ] split builder
- [ ] dedup
- [ ] task-family registry
- [ ] leakage checker

### P2

- [ ] MaleCNS loader
- [ ] central-brain filter
- [ ] community aggregation
- [ ] deterministic K=32 coarse-graining
- [ ] transition operator
- [ ] graph statistics
- [ ] graph manifest

### P3

- [ ] uniform random graph
- [ ] degree-preserving rewire
- [ ] module-preserving random graph
- [ ] null constraint validator

### P4

- [ ] backbone loader
- [ ] feature extractor
- [ ] ParameterBlock
- [ ] block injection
- [ ] block trainer
- [ ] block bank manifest

### P5

- [ ] Base + 8 blocks batch inference
- [ ] strict scorer
- [ ] candidate cache
- [ ] FIX/BREAK/UNCHANGED labeler
- [ ] Oracle report

### P6

- [ ] input projector
- [ ] fixed transfer operator
- [ ] action head
- [ ] loss
- [ ] router trainer
- [ ] router inference

### P7

- [ ] fixed coverage evaluator
- [ ] calibration
- [ ] risk metrics
- [ ] clustered bootstrap
- [ ] routing trace

完成 P0-P7 后，才能正式进入 H1。

---

## 31. 首次正式开发完成标准

第一阶段代码完成应满足：

```text
1. 一条命令可构建 MaleCNS K=32 graph
2. 一条命令可构建 matched null graphs
3. 一条命令可训练 8-block bank
4. 一条命令可生成 candidate cache
5. 一条命令可训练任一 topology router
6. 一条命令可执行 fixed-coverage evaluation
7. 一条命令可执行 calibration + deployment evaluation
8. 所有 run 自动保存 manifest
9. 所有 sample 自动保存 routing trace
10. pytest 全部通过
```

建议最终命令形式：

```bash
python scripts/build_malecns_graph.py \
  --config configs/graph/malecns_k32.yaml

python scripts/train_blocks.py \
  --config configs/block/block_bank_8.yaml

python scripts/generate_candidate_cache.py \
  --config configs/experiment/h1_main.yaml

python scripts/train_router.py \
  --graph malecns \
  --config configs/experiment/h1_main.yaml

python scripts/evaluate_h1.py \
  --config configs/experiment/h1_main.yaml
```

---

## 32. 开发阶段的唯一优先级

当前开发优先级固定为：

```text
Data correctness
>
Block headroom
>
Candidate cache correctness
>
Graph correctness
>
Router correctness
>
H1 evidence
>
H2 mechanism
>
H3 abstraction
>
System optimization
```

不要为了提前做更复杂的系统功能破坏这个顺序。

---

## 33. 当前阶段最终目标

当前阶段不要求证明 MaleCNS 一定有效。

开发完成的标准是：

> 建立一个可以公平、可重复、可审计地比较真实 MaleCNS topology 与严格匹配 null topology 的 frozen parameter-block selective repair 平台。

实验结果决定后续叙事，而不是代码结构预设结论。

