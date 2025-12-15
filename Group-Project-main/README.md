# Group Project Setup Guide

## Project Content

- Custom Environment: 基於 MountainCar-v0，新增物理特性 (Mass/Force) 與地形系統 (Boost/Mud)。
- Vehicle System: Standard (標準), SportCar (跑車), Truck (卡車)。

- Agent Brain:
   - RL Agents: Q-Learning, SARSA (具備學習能力)。
   - Baseline: Rookie (Random), Pro (Rule-Based)。

- Manager: 支援 Train (訓練), Play (展示), Compare (比較) 模式。

## Installation

```bash

# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Navigate to the Gymnasium directory
cd group_project/Gymnasium

# 4. Install Gymnasium in editable mode
pip install -e .

# 5. Install additional dependencies
pip install "gymnasium[classic_control]"
pip install matplotlib

#6. Install moviepy for video
python -m pip install moviepy
```

## ✅ Verification

Run the following command to verify that the installation is successful:

```bash
% pip list
```

Sample Output from MacOS:

```
Package              Version Editable project location
-------------------- ------- --------------------------------------------
cloudpickle          3.1.2
Farama-Notifications 0.0.4
gymnasium            1.2.2   ./group_project/Gymnasium
numpy                2.3.5
pip                  24.3.1
typing_extensions    4.15.0
```

If your output matches the above (or is similar), your environment is correctly configured.

---

## 🚀 Running the Project

### **Part 2: Frozen Lake**
Run the Frozen Lake environment:

```bash
python frozen_lake.py
```


### **Part 3: Mountain Car**
Standard Command Format
You can view the full list of arguments by running:

```bash

python mountain_car.py --help
```

Usage:

```bash
usage: mountain_carn.py [-h] (--train | --play | --compare) 
               [--agent {qlearning,sarsa,rookie,pro}]
               [--vehicle {Standard,SportCar,Truck}] 
               [--episodes EPISODES] [--record]
               [--terrain {normal,mud,hard}]
```


#### **1: Training Mode (Train)**
Train an agent and save the model (.pkl) and learning curve (.png).

```bash

# Example: Train a SportCar using Q-Learning on normal terrain
python mountain_car.py --train --agent qlearning --vehicle SportCar --episodes 1000 --terrain NoMud

# Example: Train a Truck using SARSA on hard mud terrain
python mountain_car.py --train --agent sarsa --vehicle Truck --episodes 2000 --terrain HardMud
```

#### **2: Play/Demo Mode (Play)**
Visualize the performance of a trained model.

```Bash

# Watch a trained Q-Learning SportCar
python mountain_car.py --play --agent qlearning --vehicle SportCar --terrain NoMud

# Record a video (Saved in ./videos)
python mountain_car.py --play --agent qlearning --vehicle SportCar --record
```

#### **3: Comparison Mode (Compare)**
Compare all agents (Rookie, Pro, Q-Learning, SARSA) in the same environment.

```Bash

# Compare performance in Mud terrain
python mountain_car.py --compare --vehicle Standard --terrain Mud --episodes 100
```


**Tip: If you’re on Windows, replace**

```Bash

source .venv/bin/activate
```

with

```bash

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\activate
```



## Contribution List

- 劉庭甄：Part2, Part3, Reflection Report, Presentation Slides, github environment, README.md , Report

- 賴亭羽：Part2, Part3, Reflection Report, Presentation Slides, UML diagram , Report

- 謝昀恩：Part2, Part3, Presentation Slides , Report

