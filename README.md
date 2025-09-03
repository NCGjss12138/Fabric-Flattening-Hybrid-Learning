# Fabric Flattening with Dual-Arm Manipulator via Hybrid Imitation and Reinforcement Learning

This repository provides the implementation of our paper:  
**Fabric Flattening with Dual-Arm Manipulator via Hybrid Imitation and Reinforcement Learning** (*submitted to Advanced Robotics Research*).

---

## Overview
We propose a hybrid learning framework for robotic fabric flattening with a dual-arm manipulator. The method integrates:  
- **UNet-based Proposal Network**: generates probability maps of candidate operation points.  
- **Actor-Critic Networks**: predict and evaluate operation point coordinates.  
- **Hybrid optimization**: imitation learning with human-labeled data + reinforcement learning with real-world reward (fabric area increase).  

---

## Setup
Clone repository and install dependencies:
```bash
git clone https://github.com/yourname/Fabric-Flattening-Hybrid-Learning.git
cd Fabric-Flattening-Hybrid-Learning
pip install -r requirements.txt
```
---

## Usage

### Pre-training UNet
```bash
python train_unet.py --data_dir ./dataset
```
### Actor pre-training
```bash
python train_actor.py --data_dir ./dataset
```
### Joint reinforcement learning
```bash
python train_joint.py --data_dir ./dataset
```
### Testing
```bash
python test_model.py --origin ./test/origin/ --candidate ./test/candidate/
```

## dataset
**Proprietary, partially released.** The images and labels were collected in our lab and are **not a public benchmark**. To support reproducibility, this repo includes a **small sample** under `dataset/` so you can run the scripts and inspect the I/O formats. The sample is not intended for full training/evaluation.

Sample folders:
- `origin/` – original cloth images
- `candidate/` – candidate point maps
- `label/` – human-labeled points
- `mask/` – cloth masks
- `reward/` – reward values (area increase)

**Requesting the full dataset.**  
For **non-commercial research use**, you may request access to the full dataset by emailing **ma.youchun.q7@dc.tohoku.ac.jp** with your affiliation, project title, and intended use. 


## results

- **Overall success rate**: 82% → 100%  
- **One-time success rate**: 74% → 94%  
- Generalizes to fabrics with different thicknesses and stiffnesses.

## citation
```bibtex
@article{Ma2025FabricFlattening,
  title={Fabric Flattening with Dual-Arm Manipulator via Hybrid Imitation and Reinforcement Learning},
  author={Ma, Youchun and Tokuda, Fuyuki and Seino, Akira and Kobayashi, Akinari and Kosuge, Kazuhiro},
  journal={Advanced Robotics Research},
  year={2025}
}
