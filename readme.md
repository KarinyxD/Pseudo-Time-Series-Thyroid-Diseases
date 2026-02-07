# 🧬 Thyroid Disease Trajectory: Constructing Pseudo-Time Series from Static Clinical Data

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Topic](https://img.shields.io/badge/Topic-Trajectory_Inference-orange?style=flat-square)
![Method](https://img.shields.io/badge/Method-Graph_Theory_%26_Bootstrap-blueviolet?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

## 📖 Overview
**How do you map the evolution of a disease when you only have static snapshots?**

This project implements a **Trajectory Inference** pipeline that reconstructs the continuous progression of Hypothyroidism using cross-sectional clinical data. By applying **Graph Theory (Minimum Spanning Tree)** and **Manifold Learning** concepts, we mathematically order patients along a "Pseudo-Time" axis.

The result is a dynamic model that transforms static hormonal profiles (TSH, T3, T4) into a longitudinal timeline, allowing for the study of disease transition states without requiring years of patient follow-up.

---

## 🎯 The Core Problem: The "Longitudinal Gap"
In healthcare analytics, understanding *how* a patient transitions from "Healthy" to "Severe" is crucial. However, true **Longitudinal Data** (tracking the same patient over years) is:
* Rare and expensive.
* Often unavailable for large populations.

Most available datasets are **Cross-Sectional** (snapshots of many different patients at a single point in time). Traditional Machine Learning classifiers can label these patients, but they cannot tell the story of *progression*.

### 💡 The Solution
We treat the N-dimensional feature space as a topological map. By calculating the **Geodesic Distance** (shortest path) between patients on a graph, we infer their relative position in the disease timeline.

---

## 🔬 Methodology & Architecture

The pipeline is modularized into four distinct stages:

### 1. Data Embedding & Space Construction
* **Input:** Raw clinical data (`thyroidDF.csv`).
* **Process:** Outlier removal, cleaning, and Z-Score normalization of hormonal features (TSH, T3, TT4, T4U, FTI, Age).
* **Goal:** Create a normalized feature space where Euclidean distance represents biological similarity.

### 2. Topological Modeling (MST)
* **Algorithm:** We compute a similarity matrix and build a **Minimum Spanning Tree (MST)**.
* **Why MST?** It connects all patients with the minimum possible total edge weight, revealing the "skeleton" of the data structure and filtering out noise/weak connections.

### 3. Pseudo-Time Inference (Dijkstra)
* **Root Definition:** A centroid of healthy patients is defined as $t=0$.
* **Pathfinding:** We use **Dijkstra's Algorithm** to calculate the distance from the root to every other patient along the MST edges.
* **Result:** This distance is the **Pseudo-Time**. It represents the "severity score" or the evolutionary stage of the disease.

### 4. Stochastic Validation (Bootstrap)
* To ensure the trajectory isn't an artifact of specific outliers, we perform **Bootstrap Resampling** (Monte Carlo simulation).
* **50+ Iterations:** We generate 50 parallel datasets, build 50 MSTs, and compute 50 distinct trajectories.
* **Consensus:** The final output is a robust consensus trajectory that minimizes variance.

---

## 📊 Visual Analysis

### The "Trajectory Beam" (Stability Analysis)
This visualization validates the robustness of the inferred timeline.

![Bootstrap Beam Plot](figures/Figure_8.png)
*> **Figure 1:** The "Beam" of Disease Progression. The X-axis represents the inferred **Pseudo-Time**. The Y-axis represents TSH levels. Each thin grey line is a bootstrap simulation. The **solid colored line** is the consensus trajectory. Note the exponential rise in TSH as the pseudo-time advances, capturing the biological feedback loop failure.*

### The Manifold Topology (MST)
![MST Graph](figures/Figure_7.png)
*> **Figure 2:** The Minimum Spanning Tree projection. Nodes represent patients, colored by clinical severity. The structural path from the Healthy cluster (Blue) to the Severe cluster (Red) physically demonstrates the inferred trajectory.*

---

## 📂 Project Structure

The codebase is organized as a scalable Python package:

```text
thyroid-trajectory/
├── data/
│   ├── raw/                # Original dataset (immutable)
│   ├── processed/          # Cleaned data and final CSV outputs
│   ├── results             # 
|
├── figures/                # Generated plots (High-Res)
│
├── src/                    # Source Code Package
│   ├── __init__.py         # Package initialization
│   ├── preprocessing.py    # ETL, Imputation & Normalization
│   ├── euclidean_matrix.py # Distance Matrix Computation
│   ├── mst.py              # Graph Topology & Tree Construction
│   ├── trajectory.py       # Pathfinding (Dijkstra) & Sorting Logic
│   └── bootstrap.py        # Resampling Orchestration
│
├── main.py                 # Main execution pipeline
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

---

## 🚀 How to Run

To replicate the analysis and generate the pseudo-time trajectories:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/KarinyxD/Pseudo-Time-Series-Thyroid-Diseases.git](https://github.com/seu-usuario/thyroid-trajectory.git)
    cd thyroid-trajectory
    ```

2.  **Install dependencies:**
    Ensure you have Python 3.8+ installed. Then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Inference Pipeline:**
    Execute the main script to process data, run the bootstrap, and export results:
    ```bash
    python main.py
    ```

### Output
The script will generate a CSV file in `data/results/trajectories.csv` containing:
* **Original Clinical Values:** (TSH, T3, TT4, T4U, FTI, Age).
* **Inferred Pseudo-Time:** The calculated disease progression score.
* **Bootstrap Metadata:** Tracking of stability across the 50 simulation rounds.

---

## 👤 Author

**Kariny Abrahão**
*Computer Scientist*