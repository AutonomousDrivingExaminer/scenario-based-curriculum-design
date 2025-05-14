# Scenario-Based Curriculum Generation for Multi-Agent Autonomous Driving

## Installation
```bash
git clone https://github.com/AutonomousDrivingExaminer/mats-trafficgen
cd mats-trafficgen
pip install -r requirements.txt
pip install -U hydra-core
```

## Usage
The code for training the dual-curriculum design agent is in `training/train_plr.py`.
To start a training run, execute the following command:
```bash
cd training
PYTHONPATH=.. python train_plr.py
```
The training script can be configured by modifying the config file in `training/configs/ued_route_following.yaml`.
