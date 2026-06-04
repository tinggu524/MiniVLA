<div align="center">
  <h1>MiniVLA</h1>
  <p>Build a tiny VLA from scratch, then upgrade it step by step toward modern robot policy architectures.</p>
  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-MiniVLA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
    <img alt="Toy Task" src="https://img.shields.io/badge/2D_Tabletop-Toy_Task-19A974?style=for-the-badge">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-black?style=for-the-badge">
  </p>
  <p>
    <a href="./README.md"><b>中文</b></a>
  </p>
</div>

## 📚 Table of Contents

- [🎯 Goal](#goal)
- [🧩 Model Contents](#model)
- [🗺️ Learning Roadmap](#roadmap)
- [📊 Results](#results)
- [📁 Project Structure](#structure)
- [🚀 Usage](#usage)
- [🏷️ Releases](#releases)
- [📄 License](#license)

<a id="goal"></a>
## 🎯 Goal

MiniVLA is a learning-roadmap project for people who want to understand Vision-Language-Action models from the ground up. It uses a small 2D tabletop toy task to build the full loop:

```text
generate expert demos -> train behavior cloning -> rollout evaluation -> failure analysis -> component upgrades
```

The goal is not to reproduce OpenVLA, ACT, or π0 immediately. The goal is to first build the smallest working VLA skeleton, then upgrade one component at a time and measure whether the change actually helps.

<a id="model"></a>
## 🧩 Model Contents

v0 is the minimal runnable version:

```text
image        -> 3-layer CNN
instruction  -> small vocab + embedding + mean pooling
state        -> 2-layer MLP
fusion       -> concat + MLP
action head  -> 2-layer MLP
loss         -> MSE
```

Inputs:

```text
64x64 RGB image
language instruction: "move red block to green target"
robot state: [gripper_x, gripper_y, is_holding]
```

Output:

```text
action: [dx, dy, gripper_action]
```

The task is a simplified 2D tabletop pick-and-place environment:

```text
move the blue gripper to the red block
close the gripper
move the red block to the green target
open the gripper
check success
```

<a id="roadmap"></a>
## 🗺️ Learning Roadmap

The project will upgrade components step by step. Each version should be trained, evaluated, and compared.

| Version | Upgrade | What to Learn |
|---|---|---|
| v0 | 3-layer CNN + small vocab + MLP fusion + MSE | Minimal VLA loop |
| v1 | action chunking | Why ACT predicts more than one action step |
| v2 | transformer fusion | Move from concat fusion to token-level fusion |
| v3 | transformer decoder action head | Generate future actions with action queries |
| v4 | ResNet18 vision encoder | Test whether a stronger vision backbone improves generalization |
| v5 | CLIP/SigLIP-style encoder | Use pretrained vision-language representations |
| v6 | multiple colors, objects, and instructions | Make language grounding matter |
| v7 | diffusion / flow action head | Connect to modern continuous-action VLA policies |
| v8 | compare ACT, OpenVLA, π0, and SmolVLA | Map toy lessons to real architectures |

Each version should record:

```text
train loss
success rate
mean final distance
failure GIFs
success example GIFs
parameter count
main failure modes
```

<a id="results"></a>
## 📊 Results

Latest v0 evaluation:

```text
success rate: 0.94
mean final distance: 3.07
```

The result is saved in:

```text
outputs/v0/results_v0.json
```

The evaluation script saves failure GIFs and a few success examples, but generated GIFs are ignored by Git.

<a id="structure"></a>
## 📁 Project Structure

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
## 🚀 Usage

Create the environment:

```bash
conda create -n minivla python=3.10
conda activate minivla
python -m pip install numpy imageio torch torchvision
```

Generate demos:

```bash
python minivla_data.py
```

Train v0:

```bash
python train_v0.py
```

Evaluate v0:

```bash
python eval_v0.py
```

Evaluation outputs:

```text
outputs/v0/results_v0.json
outputs/v0/rollouts/failures/
outputs/v0/rollouts/success_examples/
```

<a id="releases"></a>
## 🏷️ Releases

When a stable milestone is complete, create a GitHub Release, for example:

```text
v0.1.0  MiniVLA baseline
v0.2.0  action chunking
v0.3.0  transformer fusion
```

Releases are useful for stable milestones, experiment summaries, key metrics, and downloadable artifacts. Regular development should still be tracked with commits. When a version is polished enough to show, create a tag and publish a release.

<a id="license"></a>
## 📄 License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.
