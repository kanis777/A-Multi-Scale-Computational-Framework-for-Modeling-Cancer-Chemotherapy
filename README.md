# 🧬 A Multi-Scale Computational Framework for Modeling Cancer Chemotherapy

This repository contains the **Python code** for a research project that models the dynamic battle between the **immune system**, **cancer cells**, and **chemotherapy**.  

The project's goal is to build a more realistic cancer simulation. It starts with a simple, standard mathematical model (from a 2019 paper by *Lestari et al.*) and builds on it in two major steps—creating a final model that is far more **biologically accurate**.

---

## 🎯 Research Ideology: A Three-Model Progression

Our research is built on the idea that **simple models have critical limitations**.  
Real-world cancer is not just an "average" calculation—it's a **complex, random, and spatial battle**.

We developed **three models** to capture this progression:

---

### 🔹 Model 1: The Baseline — *"The Calculator"*

**Concept:**  
This is the starting point, based on *Lestari et al. (2019)*. It treats the three populations —  
- Immune Cells (**E**)  
- Tumor Cells (**T**)  
- Chemotherapy (**M**)  
as large, perfectly mixed “pots” of liquid.

**How it Works:**  
- Uses standard **ordinary differential equations (ODEs)** to calculate the average outcome.  
- Running the simulation 100 times produces the **exact same result** each time.

**Limitation:**  
- Not random (deterministic).  
- Has **no spatial awareness** — cannot distinguish between a solid tumor and a well-mixed cell suspension.

---

### 🔹 Model 2: The Stochastic Model — *"The Forecast"*

**Concept:**  
This model introduces **biological randomness (stochasticity)**.

**How it Works:**  
- Converts the “calculator” into a **forecast** using a **Wiener Process**.  
- At every small time step, the model “rolls the dice” to simulate random biological variations.  
- Implemented using the **Euler-Maruyama method**.

**What it Gives Us:**  
- Produces a **distribution of possible futures** rather than a single outcome.  
- Allows researchers to ask questions like:  
  > “What is the probability this treatment will work?”

**Limitation:**  
- Still assumes a single, well-mixed tumor — **no spatial structure**.

---

### 🔹 Model 3: The Network-Hybrid Model — *"The Battle Simulator"*

**Concept:**  
The most advanced model, adding **spatial structure** to the simulation.

**How it Works:**  
- Uses **NetworkX** to represent the tumor as a **graph** of interconnected cell agents.  
- Each node represents a tumor cell; edges represent physical proximity or communication.

**New, Smarter Biological Rules:**

- **Immune Attack (Surface vs. Core):**  
  Immune cells can only attack the **surface cells** (nodes with fewer connections).  
  Core cells are shielded from direct attack.

- **Chemo Penetration (Gradient):**  
  Chemotherapy originates from a **source node** (representing a blood vessel) and weakens with **distance** (via shortest path length).

- **Tumor Growth (Agent-Based):**  
  Tumor cells divide and add **new nodes** to the network, simulating physical tumor expansion.

---

## 📁 Python Files Overview

| File | Model | Description |
|------|--------|-------------|
| **`orig.py`** | Model 1 | Implements the baseline deterministic model from Lestari et al. (2019). Uses SciPy to solve ODEs for a single "average" result. |
| **`ver2_sde.py`** | Model 2 | Implements the stochastic "forecast" model. Defines drift and diffusion functions and includes a custom SDE solver using the Euler-Maruyama method. |
| **`ver3_hybrid.py`** | Model 3 | Implements the advanced Network-Hybrid model ("battle simulator") with spatial logic, immune attack surface mechanics, chemo gradients, and agent-based tumor growth. |

---

## 🔍 Key Finding

The **power of this multi-scale approach** is revealed in simulation results (see `NS_PACKAGE_slide.pdf`).

### 🧪 Scenario 1: Low Tumor Growth, Low Chemo
| Model | Prediction | Outcome |
|--------|-------------|----------|
| **Model 1 (ODE)** | Tumor eliminated | ❌ Unrealistic |
| **Model 2 (SDE)** | Tumor eliminated | ❌ Unrealistic |
| **Model 3 (Network-Hybrid)** | Tumor survives | ✅ Realistic |

**Why the Difference?**  
> **Spatial protection.**  
> In the Network-Hybrid model, immune cells can only kill the **surface** tumor cells.  
> The **core** is shielded — allowing the tumor to persist.  
> This reproduces realistic **treatment resistance**, which simpler models cannot capture.

---

## 📚 Reference

> Lestari, D., Sari, E. R., & Arifah, H. (2019).  
> *Dynamics of a mathematical model of cancer cells with chemotherapy.*  
> *Journal of Physics: Conference Series, 1320, 012026.*

---

## ⚙️ Requirements

- Python ≥ 3.8  
- `numpy`  
- `scipy`  
- `networkx`  
- `matplotlib`  

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/kanis777/A-Multi-Scale-Computational-Framework-for-Modeling-Cancer-Chemotherapy.git

# Run the baseline model
python orig.py

# Run the stochastic model
python ver2_sde.py

# Run the network-hybrid model
python ver3_hybrid.py

or run the .ipynb file
