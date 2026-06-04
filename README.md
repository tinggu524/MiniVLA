<div align="center">
  <h1>MiniVLA</h1>
  <p>从零开始搭建一个极小 VLA，并逐步升级到现代机器人策略架构。</p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-miniVLA-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/task-2D%20tabletop-19a974" alt="Toy Task">
  <img src="https://img.shields.io/badge/license-MIT-black" alt="License">
</p>

<p align="center">
  <b>中文</b> ｜ <a href="./README.en.md">English</a>
</p>

## 📚 目录

- [🎯 项目目标](#goal)
- [🧩 模型内容](#model)
- [🗺️ 学习路线](#roadmap)
- [📊 实验结果](#results)
- [📁 项目结构](#structure)
- [🚀 运行方式](#usage)
- [📄 License](#license)

<a id="goal"></a>
## 🎯 项目目标

MiniVLA 是一个面向 VLA 初学者的学习路线项目。它用一个 2D tabletop toy task 搭建完整闭环：

```text
生成专家数据 -> 训练行为克隆模型 -> rollout 评估 -> 分析失败案例 -> 升级组件
```

这个项目的核心不是一开始就复现 OpenVLA、ACT 或 π0，而是先把 VLA 最小骨架跑通，再逐个升级组件，理解每个模块为什么存在、解决什么问题、是否真的改善效果。

<a id="model"></a>
## 🧩 模型内容

v0 是一个最小可运行版本：

```text
image        -> 3-layer CNN
instruction  -> small vocab + embedding + mean pooling
state        -> 2-layer MLP
fusion       -> concat + MLP
action head  -> 2-layer MLP
loss         -> MSE
```

输入：

```text
64x64 RGB image
language instruction: "move red block to green target"
robot state: [gripper_x, gripper_y, is_holding]
```

输出：

```text
action: [dx, dy, gripper_action]
```

任务是一个简化的 2D 桌面抓取：

```text
蓝色夹爪移动到红色方块
闭合夹爪
把红色方块移动到绿色目标区
打开夹爪
判断是否成功
```

<a id="roadmap"></a>
## 🗺️ 学习路线

这个仓库会按组件逐步升级，每一步都训练、评估并记录结果。

| 版本 | 升级点 | 学习重点 |
|---|---|---|
| v0 | 3-layer CNN + small vocab + MLP fusion + MSE | 最小 VLA 闭环 |
| v1 | action chunking | 理解 ACT 为什么不只预测单步动作 |
| v2 | transformer fusion | 从 concat 融合升级到 token-level 融合 |
| v3 | transformer decoder action head | 用 action queries 生成未来动作序列 |
| v4 | ResNet18 vision encoder | 观察视觉 backbone 升级是否提高泛化 |
| v5 | CLIP/SigLIP-style encoder | 接入预训练视觉语言表征 |
| v6 | 多颜色、多物体、多指令 | 让语言 grounding 真正变重要 |
| v7 | diffusion / flow action head | 对齐现代 VLA 的连续动作生成路线 |
| v8 | 对比 ACT、OpenVLA、π0、SmolVLA | 把 toy 经验映射到真实架构 |

每个版本都应该记录：

```text
train loss
success rate
mean final distance
失败 GIF
成功样例 GIF
参数量
主要失败原因
```

<a id="results"></a>
## 📊 实验结果

最近一次 v0 评估：

```text
success rate: 0.94
mean final distance: 3.07
```

结果保存在：

```text
outputs/v0/results_v0.json
```

评估脚本会保存失败案例 GIF 和少量成功样例 GIF，但这些生成文件默认不会进入 Git。

<a id="structure"></a>
## 📁 项目结构

```text
MiniVLA/
  minivla_data.py       # toy environment, expert policy, dataset
  minivla_model.py      # MiniVLA model
  train_v0.py           # train v0 with MSE behavior cloning
  eval_v0.py            # rollout evaluation and failure analysis
  data/                 # generated .npz demos, ignored by Git
  checkpoints/          # generated model weights, ignored by Git
  outputs/v0/           # metrics tracked, GIFs ignored by Git
```

<a id="usage"></a>
## 🚀 运行方式

创建环境：

```bash
conda create -n minivla python=3.10
conda activate minivla
python -m pip install numpy imageio torch torchvision
```

生成数据：

```bash
python minivla_data.py
```

训练 v0：

```bash
python train_v0.py
```

评估 v0：

```bash
python eval_v0.py
```

评估后会生成：

```text
outputs/v0/results_v0.json
outputs/v0/rollouts/failures/
outputs/v0/rollouts/success_examples/
```

<a id="license"></a>
## 📄 License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.
