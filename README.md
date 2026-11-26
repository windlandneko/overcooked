# Overcooked 零样本协作

强化学习课程团队任务的纯强化学习策略：每张图训练一个循环神经网络，提交包为纯正的前向推理。

> 平台会把每组的提交与其他组配对，我们的策略需要能和没见过的搭档合作。

## 环境要求

- [uv](https://docs.astral.sh/uv/)：自动安装 Python 3.12 和依赖（和平台的 torch 2.14.1 一致）
- 编译器需要支持 C++20：第一次导入 `zsc.engine` 时会编译我们自己的超光速 C++ 模拟引擎

## 快速开始

```bash
git clone https://github.com/windlandneko/overcooked.git
cd overcooked
uv sync
```

将课程提供的伙伴包放在 `data/teachers/fcp` 和 `data/teachers/mep`；所有训练产物最终会写入 `runs/`。

## 训练

按顺序运行以下步骤：

```bash
# 1. 把伙伴包按 seed 分为 train、dev 两段
uv run -m zsc.teachers

# 2. 训练 16 种风格的 CNN 搭档和 7 个模板 MLP 搭档
uv run -m zsc.population

# 3. 模拟其他组的训练方式，生成同学搭档
uv run -m zsc.classmates --group mlp
uv run -m zsc.classmates --group kit
uv run -m zsc.classmates --evaluate

# 4. 每张图训练一个学生策略
uv run -m zsc.student --teachers runs/teacher_pack --classmates runs/classmates --min-partner-sp 20 --output runs/students

# 5. 量化为 int8 并打包提交
uv run -m zsc.package runs/students/{unident_s_hard,counter_circuit,schelling} --output runs/submission

# 查看实时训练曲线
uv run tensorboard --logdir runs
```

| 步骤 | 输出位置 |
| --- | --- |
| `zsc.teachers` | `runs/teacher_pack` |
| `zsc.population` | `runs/population` |
| `zsc.classmates` | `runs/classmates` |
| `zsc.student` | `runs/students/<地图>` |
| `zsc.package` | `runs/submission` |

如果想从已训练的学生继续微调，只需给 `zsc.student` 加上 `--init-from <目录>`。：

## 评测

```bash
# 单元测试（引擎与课程环境逐位一致、伙伴包、int8 精度、策略记忆等）
uv run pytest

# 与伙伴包的 dev 成员配合，两个座位都评测一遍
uv run -m zsc.evaluate runs/submission/agent

# 与模拟同学配合
uv run -m zsc.evaluate runs/submission/agent --partners classmates --layouts counter_circuit

# 课程的提交检查
uv run check.py runs/submission/submission.zip --trusted-own-code --episodes 5
```

## 注意事项

- test 段只用于最终汇报：伙伴包成员按 seed 分段，模拟同学按整个训练 run 分段。训练和选模只读 train、dev。`runs/teacher_pack` 里没有 test 成员，加载 test 会直接报错。
- dev 分数噪声大：每张图 32 局时，fp32 与 int8 这样几乎相同的两个模型也能差 5 分。所以比较模型得用更多局数，或者干脆直接比较动作是否一致。
- 5 MiB 上限可以放心量化，性能损失非常小：每个权重矩阵按输出行量化成 int8，三张图共 4.0 MiB，而且动作与 fp32 一致 99.3% 以上。
- 平台可能不传 `previous_joint_action`（存疑，不过保险起见）：这时策略从前后两帧网格读出搭档的动作，记忆照常保留，只在新一局开始时清空。
- 瓶颈现在只在网络上，环境已经超光速了喵：C++ 引擎每秒约 1600w+ 步，只占训练时间的 0.1%，82% 的时间都用在 PPO 更新，其中几乎全是学生的卷积主干。

## 目录结构

```text
zsc/
  engine.py, engine.cpp   C++ 向量化环境
  policy.py               提交的网络与推理
  models.py               训练用网络
  teachers.py             伙伴包分段
  population.py           CNN 与 MLP 搭档种群
  classmates.py           模拟同学
  student.py              学生训练与微调
  package.py              int8 打包
  evaluate.py, bench.py   评测与环境测速
tests/                    单元测试
core/, agent/, train.py, check.py, export.py, config*.json, requirements.txt
                          课程套件
```

课程套件中的 `core/` 加快了观测编码和状态复制，经过多轮验证和修复可以保证规则不变。`train.py` 是课程模板的多进程 PPO 自对弈训练器，默认使用我们的 C++ 引擎，提交或者 debug 时可以通过 `--simulator python` 切换回课程的 Python 原版环境。

## 参考文献

### 环境

| 文献 | 借鉴点 | 对应代码 |
| --- | --- | --- |
| **Overcooked-AI** — Carroll et al., *On the Utility of Learning about Humans for Human-AI Coordination*, NeurIPS 2019 ([arXiv:1910.05789](https://arxiv.org/abs/1910.05789)) | 双人协作厨房环境与规则 | `core/env.py`, `zsc/engine.cpp` |

### 搭档构建

| 文献 | 借鉴点 | 对应代码 |
| --- | --- | --- |
| **FCP** — Strouse et al., *Collaborating with Humans without Human Data*, NeurIPS 2021 ([arXiv:2110.08176](https://arxiv.org/abs/2110.08176)) | 多阶段检查点快照（`init/mid/best`、`early/mid/final`） | `zsc/teachers.py:L59-L80`, `zsc/population.py:L186-L187` |
| **MEP** — Zhao et al., *Maximum Entropy Population-Based Training for Zero-Shot Human-AI Coordination*, AAAI 2023 ([arXiv:2112.11701](https://arxiv.org/abs/2112.11701)) | 最大熵种群伙伴包接入 | `zsc/teachers.py:L59-L80` |
| **HSP** — Yu et al., *Learning Zero-Shot Cooperation with Humans, Assuming Humans Are Biased*, ICLR 2023 ([arXiv:2302.01605](https://arxiv.org/abs/2302.01605)) | 按事件加权的风格塑形奖励 | `zsc/population.py:L69-L112,L270-L280` |
| **LIPO** — Charakorn et al., *Generating Diverse Cooperative Agents by Learning Incompatible Policies*, ICLR 2023 ([OpenReview](https://openreview.net/forum?id=v0WbQ4t288)) | 跨风格交叉对局不兼容惩罚 | `zsc/population.py:L231-L232,L272-L280` |
| **UPD** — Ruhdorfer et al., *Unsupervised Partner Design Enables Robust Ad-hoc Teamwork*, 2025 ([arXiv:2508.06336](https://arxiv.org/abs/2508.06336)) | 策略动作扰动搭档（`UPD_MIX` 与跨池噪声） | `zsc/student.py:L588-L602,L678-L695,L769` |

### 训练课程

| 文献 | 借鉴点 | 对应代码 |
| --- | --- | --- |
| **Other-Play** — Hu et al., *"Other-Play" for Zero-Shot Coordination*, ICML 2020 ([arXiv:2003.02979](https://arxiv.org/abs/2003.02979)) | `schelling` 180° 对称视角增强 | `zsc/models.py:L45-L53`, `zsc/student.py:L670` |
| **PLR** — Jiang et al., *Prioritized Level Replay*, ICML 2021 ([arXiv:2010.03934](https://arxiv.org/abs/2010.03934)) | 按 regret 优先抽取弱项搭档 | `zsc/student.py:L555-L578` |
| **OvercookedV2** — Gessler et al., *OvercookedV2: Rethinking Overcooked for Zero-Shot Coordination*, ICLR 2025 ([arXiv:2503.17821](https://arxiv.org/abs/2503.17821)) | 回放池 + 合法合成残局状态增强 | `zsc/engine.cpp:L302-L385`, `zsc/student.py:L607,L706` |

### 局内适应

| 文献 | 借鉴点 | 对应代码 |
| --- | --- | --- |
| **E3T** — Yan et al., *An Efficient End-to-End Training Approach for Zero-Shot Human-AI Coordination*, NeurIPS 2023 ([OpenReview](https://openreview.net/forum?id=V2g09Xl3Q7)) | 搭档下一动作预测辅助头与混合扰动 | `zsc/policy.py:L125-L133,L177-L194`, `zsc/student.py:L888-L889` |
| **Latent Partner Strategies** — Li et al., *Modeling Latent Partner Strategies for Adaptive Zero-Shot Human-Agent Collaboration*, 2025 ([arXiv:2507.05244](https://arxiv.org/abs/2507.05244)) | 潜在搭档类型建模与 Residual-MoE 条件决策 | `zsc/policy.py:L125-L137,L177-L194` |
| **Fixed-Share** — Herbster & Warmuth, *Tracking the Best Expert*, Machine Learning 32(2), 1998 ([DOI](https://doi.org/10.1023/A:1007424614876)) | 在线贝叶斯信念更新与防锁死转移 | `zsc/policy.py:L55-L75,L184` |

### 优化算法

| 文献 | 借鉴点 | 对应代码 |
| --- | --- | --- |
| **PPO** — Schulman et al., *Proximal Policy Optimization Algorithms*, 2017 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)) | 截断代理目标与价值裁剪 | `zsc/student.py:L880-L889` |
| **GAE** — Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016 ([arXiv:1506.02438](https://arxiv.org/abs/1506.02438)) | 广义优势估计 | `zsc/models.py:L33-L42`, `zsc/student.py:L847` |
| **Asymmetric Actor-Critic** — Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning*, RSS 2018 ([arXiv:1710.06542](https://arxiv.org/abs/1710.06542)) | 训练期 CTDE 特权信息 Critic（推理剥离） | `zsc/models.py:L56-L112` |

### 网络结构

| 文献 | 借鉴点 | 对应代码 |
| --- | --- | --- |
| **CoordConv** — Liu et al., *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution*, NeurIPS 2018 ([arXiv:1807.03247](https://arxiv.org/abs/1807.03247)) | 空间坐标通道拼接 | `zsc/policy.py:L35-L39,L139-L143`, `zsc/models.py:L151-L152` |
| **SE-Net** — Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018 ([arXiv:1709.01507](https://arxiv.org/abs/1709.01507)) | 通道注意力残差块 | `zsc/policy.py:L78-L93` |
