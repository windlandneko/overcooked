# Overcooked 零样本协作

强化学习课程团队任务的纯强化学习策略：每张图训练一个循环神经网络，提交包为纯前向推理。

> 平台会把每组的提交与其他组配对，策略需要能和未见过的搭档合作。

## 环境要求

- [uv](https://docs.astral.sh/uv/)：自动安装 Python 3.12 和依赖（与平台的 torch 2.14.1 一致）
- 支持 C++20 的编译器：首次导入 `zsc.engine` 时自动编译 C++ 向量化模拟引擎

## 快速开始

```bash
git clone https://github.com/windlandneko/overcooked.git
cd overcooked
uv sync
```

然后将课程提供的伙伴包解压到 `data/teachers/fcp` 和 `data/teachers/mep`。

## 训练

按顺序运行以下步骤：

```bash
# 1. 把伙伴包按 seed 分为 train、dev 两段
uv run -m zsc.teachers

# 2. 训练 16 种 HSP 风格 CNN 搭档
uv run -m zsc.population

# 3. 每张图单阶段训练学生策略（融合伙伴包 + HSP 种群 + 8 个朴素规则搭档 + EMA 副本）
uv run -m zsc.student --teachers runs/teacher_pack --min-partner-sp 20 --output runs/students

# 4. 量化为 int8 并打包提交
uv run -m zsc.package runs/students/{unident_s_hard,counter_circuit,schelling} --output runs/submission

# 查看实时训练曲线
uv run tensorboard --logdir runs
```

| 步骤               | 输出位置                 |
| ------------------ | ------------------------ |
| `zsc.teachers`   | `runs/teacher_pack`    |
| `zsc.population` | `runs/population`      |
| `zsc.student`    | `runs/students/<地图>` |
| `zsc.package`    | `runs/submission`      |

## 评测

```bash
# 单元测试（引擎逐位一致、伙伴包、朴素搭档、GRU 分段等价与 int8 精度）
uv run pytest

# 与伙伴包 dev 成员及朴素规则搭档配合，双座位评测
uv run -m zsc.evaluate runs/submission/agent --partners teachers scripted

# 课程的提交检查
uv run check.py runs/submission/submission.zip --trusted-own-code --episodes 5
```

## 注意事项

- test 段只用于最终汇报：伙伴包成员按 seed 分段，训练和选模只读 train、dev。`runs/teacher_pack` 不含 test 成员，加载 test 会直接报错。
- dev 分数存在采样噪声：每张图 32 局时，几乎等价的两个模型也可能相差 5 分，最终候选检查点会在 `final-dev-episodes` 上重评再定稿。
- 5 MiB 上限通过逐输出行 int8 量化满足：三张图打包后低于 4 MiB，与 fp32 动作一致率在 99% 以上。
- C++ 引擎每秒可达 1600 万步，训练耗时集中在神经网络前向与反向更新。

## 目录结构

```text
zsc/
  engine.py, engine.cpp   C++ 向量化环境
  policy.py               提交的网络与推理
  models.py               训练用网络与特权 Critic
  teachers.py             伙伴包分段
  population.py           HSP CNN 搭档种群
  scripted.py             8 种朴素规则搭档（仅用于训练与评测）
  student.py              单阶段学生训练与选模
  package.py              int8 量化与打包
  evaluate.py, bench.py   评测与环境测速
tests/                    单元测试
core/, agent/, train.py, check.py, export.py, config*.json, requirements.txt
                          课程套件
```

课程套件中的 `core/` 优化了观测编码和状态复制并保持规则一致。`train.py` 默认使用 C++ 引擎，调试时可通过 `--simulator python` 切回原版 Python 环境。

## 参考文献

### 环境

| 文献                                                                                                                                                                                 | 借鉴点                 | 对应代码                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------- | ----------------------------------- |
| **Overcooked-AI** — Carroll et al., *On the Utility of Learning about Humans for Human-AI Coordination*, NeurIPS 2019 ([arXiv:1910.05789](https://arxiv.org/abs/1910.05789)) | 双人协作厨房环境与规则 | `core/env.py`, `zsc/engine.cpp` |

### 搭档构建

| 文献                                                                                                                                                                             | 借鉴点                                                     | 对应代码                                                     |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| **FCP** — Strouse et al., *Collaborating with Humans without Human Data*, NeurIPS 2021 ([arXiv:2110.08176](https://arxiv.org/abs/2110.08176))                            | 多阶段检查点快照（`init/mid/best`、`early/mid/final`） | `teachers.build_pack`, `population.train_layout` |
| **MEP** — Zhao et al., *Maximum Entropy Population-Based Training for Zero-Shot Human-AI Coordination*, AAAI 2023 ([arXiv:2112.11701](https://arxiv.org/abs/2112.11701)) | 最大熵种群伙伴包接入                                       | `teachers.build_pack` |
| **HSP** — Yu et al., *Learning Zero-Shot Cooperation with Humans, Assuming Humans Are Biased*, ICLR 2023 ([arXiv:2302.01605](https://arxiv.org/abs/2302.01605))          | 按事件加权的风格塑形奖励                                   | `population.default_styles`, `population.train_layout`                      |

### 训练课程

| 文献                                                                                                                                                                          | 借鉴点                        | 对应代码                                                   |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ---------------------------------------------------------- |
| **PLR** — Jiang et al., *Prioritized Level Replay*, ICML 2021 ([arXiv:2010.03934](https://arxiv.org/abs/2010.03934))                                                  | 按 regret 优先抽取弱项搭档    | `student.train_map` 中的 `sample_partners`                               |
| **OvercookedV2** — Gessler et al., *OvercookedV2: Rethinking Overcooked for Zero-Shot Coordination*, ICLR 2025 ([arXiv:2503.17821](https://arxiv.org/abs/2503.17821)) | 回放池 + 合法合成残局状态增强 | `engine.cpp` 的 `randomize`, `VecEnv.record_states` |

### 局内适应

| 文献                                                                                                                                                                                   | 借鉴点                               | 对应代码                                                     |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| **E3T** — Yan et al., *An Efficient End-to-End Training Approach for Zero-Shot Human-AI Coordination*, NeurIPS 2023 ([OpenReview](https://openreview.net/forum?id=6ePsuwXUwf)) | 搭档下一动作预测辅助头与条件策略输入 | `ZSCStudentNet.partner_head`, `student.train_map` 中的 `aux_loss` |

### 优化算法

| 文献                                                                                                                                                                              | 借鉴点                                  | 对应代码                                           |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | -------------------------------------------------- |
| **PPO** — Schulman et al., *Proximal Policy Optimization Algorithms*, 2017 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347))                                         | 截断代理目标与广义优势更新              | `student.train_map`                       |
| **GAE** — Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016 ([arXiv:1506.02438](https://arxiv.org/abs/1506.02438)) | 广义优势估计                            | `models.gae` |
| **Asymmetric Actor-Critic** — Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning*, RSS 2018 ([arXiv:1710.06542](https://arxiv.org/abs/1710.06542))     | 训练期 CTDE 特权信息 Critic（推理剥离） | `models.PrivilegedCriticNet`, `models.extract_privileged_obs_feats`                         |

### 网络结构

| 文献                                                                                                                                                                                         | 借鉴点           | 对应代码                                                         |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ---------------------------------------------------------------- |
| **CoordConv** — Liu et al., *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution*, NeurIPS 2018 ([arXiv:1807.03247](https://arxiv.org/abs/1807.03247)) | 空间坐标通道拼接 | `policy.coord_grid`, `ZSCStudentNet.encode_spatial`, `models.PartnerNet` |
| **SE-Net** — Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018 ([arXiv:1709.01507](https://arxiv.org/abs/1709.01507))                                                          | 通道注意力残差块 | `policy.SEResBlock`                                        |

### 已实测并移除的方案（简录）

以下文献与模块曾在早期版本实现并验证，因边际收益不显著、出现空桶/刷分或与主线重叠而从代码中移除：

| 文献 / 模块 | 原机制 | 移除依据 |
| --- | --- | --- |
| **Latent Partner Strategies** — Li et al., *Modeling Latent Partner Strategies for Adaptive Zero-Shot Human-Agent Collaboration*, 2025 ([arXiv:2507.05244](https://arxiv.org/abs/2507.05244)) | `K=12` 手工搭档原型分类 + `actor_moe` 残差头 | 三张图测试老师上 12 个桶均有约 8 个桶空置，真实 RL 搭档不服从手工离散分桶；简化为 2 层 GRU + 单一 6 维 `partner_head` 后包体更小（`3.94 MiB`）且表现不降反升 |
| **Fixed-Share** — Herbster & Warmuth, *Tracking the Best Expert*, Machine Learning 32(2), 1998 ([DOI](https://doi.org/10.1023/A:1007424614876)) | 在线贝叶斯信念更新 `update_belief` 与 `eta` 网格调参 | 依赖上述 `K=12` 离散桶，移去后省掉每步信念更新与 `eta` 网格搜索 |
| **LIPO** — Charakorn et al., *Generating Diverse Cooperative Agents by Learning Incompatible Policies*, ICLR 2023 ([OpenReview](https://openreview.net/forum?id=BHKGTogucl)) | Stage 1 跨风格 16% 交叉对局不兼容负惩罚 | 在 `counter_circuit` 等稀疏图上与自身策略学习互相干扰；纯 HSP 塑形更稳且产出更高自博弈分的搭档 |
| **UPD** — Ruhdorfer et al., *Unsupervised Partner Design Enables Robust Ad-hoc Teamwork*, 2025 ([arXiv:2508.06336](https://arxiv.org/abs/2508.06336)) | `UPD_MIX` 与跨池 Dirichlet 动作扰动 | 随机动作噪声偏离真实同学与死板脚本行为，改用 8 个朴素规则搭档（`zsc/scripted.py`）替代后死锁与下限显著提升 |
| **Other-Play** — Hu et al., *"Other-Play" for Zero-Shot Coordination*, ICML 2020 ([arXiv:2003.02979](https://arxiv.org/abs/2003.02979)) | `schelling` 180° 旋转视角与动作置换增强（`ROT180`） | C++ 引擎采样吞吐高（约 1600 万步/秒），双座位真实样本充足；移除通道置换逻辑降低维护风险且不损 `schelling` 成绩 |
| 模拟同学池与二阶段热启 | `zsc/classmates.py`（659 行）及 `--init-from` 两段式微调 | 与 HSP 种群和官方伙伴包高度同质重叠；单阶段联合 `teachers + HSP + scripted` 训练即追平并反超旧二阶段模型 |
| 模板 MLP 搭档、历史副本与温度网格 | `MLPPartnerNet` 家族、`SP_HIST1/2` 双历史快照、`meta.json` 温度搜索 | `counter_circuit` 上 MLP 全为 0 分且另两图与 `early/mid` CNN 与老师重叠；`SP_EMA` 单副本即可维持自博弈；实测 `temp=0.0`（确定性 `argmax`）与搜索最优仅差 `0~0.2` 分（纯噪声） |
