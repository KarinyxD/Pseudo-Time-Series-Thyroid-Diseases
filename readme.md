# 🧬 Thyroid Disease Trajectory: Constructing Pseudo-Time Series from Static Clinical Data

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Topic](https://img.shields.io/badge/Topic-Trajectory_Inference-orange?style=flat-square)
![Method](https://img.shields.io/badge/Method-Graph_Theory_%26_Bootstrap-blueviolet?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

## 📖 Overview
**How do you map the temporal evolution of a disease using only static snapshots?**

This project implements a **Trajectory Inference** pipeline designed to construct **Pseudo-Time Series** from static **cross-sectional databases**.

Traditional cross-sectional studies provide only a snapshot of disease processes, lacking the temporal dimension required for prognostic modeling. To bridge this gap, we apply **Euclidean Distance** and **Graph Theory (Minimum Spanning Tree)** to reconstruct the latent temporal structure of Hypothyroidism.

By mathematically ordering patients along an inferred **"Pseudo-Time" axis**, we transform static hormonal profiles (TSH, T3, T4, FTI, T4U) into a continuous longitudinal timeline. This allows for the analysis of disease transition states and progression dynamics without the prohibitive cost and time constraints of collecting true longitudinal data.

---

## 🎯 The Core Problem: The "Longitudinal Gap"
In healthcare analytics, understanding *how* a patient transitions from "Healthy" to "Severe" is crucial. However, true **Longitudinal Data** (tracking the same patient over years) is:
* Rare and expensive.
* Often unavailable for large populations.

Most available datasets are **Cross-Sectional** (snapshots of many different patients at a single point in time). Traditional Machine Learning classifiers can label these patients, but they cannot tell the story of *progression*.

### 💡 The Solution
We model the N-dimensional feature space by computing Euclidean Distances to build a Minimum Spanning Tree (MST) structure. By defining a "Healthy" reference point as the root, we calculate the cumulative path distance along the tree to every other patient. This allows us to mathematically infer their temporal ordering and reconstruct the disease trajectory.

---

## 🔬 Methodology & Architecture

The pipeline is modularized into four distinct stages:

### 1. Data Embedding & Space Construction
* **Input:** Raw clinical data sourced from Kaggle ([Thyroid Disease Data](https://www.kaggle.com/datasets/emmanuelfwerr/thyroid-disease-data)). 
* **Process:** Outlier removal, cleaning, and Z-Score normalization of hormonal features (TSH, T3, TT4, T4U, FTI, Age).
* **Goal:** Preprocess and standardize raw data to ensure the dataset is clean, consistent, and ready for computational modeling.

### 2. Robustness Analysis (Stratified Stochastic Subsampling)
* To validate the stability of the inferred trajectory, we implemented a **Hybrid Monte Carlo Subsampling** strategy ($k=1500$ iterations).
* **Full Spectrum Coverage:** Unlike standard random sampling, which might exclude rare severe cases in imbalanced datasets (resulting in biologically truncated trajectories), we enforce **stochastic quotas** to ensure every sample covers the full disease progression ($T=30$):
    * **Severe ($n \in [1, 10]$):** Guarantees that the trajectory always reaches the disease endpoint, testing resilience in both sparse (single-case) and dense scenarios.
    * **Moderate ($n \in [5, 10]$) & Healthy ($n \in [1, 4]$):** Ensures a stable "root" and the transitional bridge between health and disease.

> **Note on Sampling Strategy:** The specific quota parameters (e.g., 1-10 severe cases) were empirically tuned to guarantee manifold connectivity for this specific cohort. Further investigation into adaptive density estimation or alternative filling strategies could refine this step, potentially reducing the need for manual constraint tuning in future applications.

### 3. Topological Modeling (Euclidean Matrix and MST)
* **Algorithm:** We compute a euclidean distance matrix and build a **Minimum Spanning Tree (MST)**.
* **Why MST?** It connects all patients with the minimum possible total edge weight, revealing the "skeleton" of the data structure and filtering out noise/weak connections.

### 4. Pseudo-Time Inference (Dijkstra)
* **Root Definition:** A centroid of healthy patients is defined as $t=0$.
* **Pathfinding:** We use **Dijkstra's Algorithm** to calculate the distance from the root to every other patient along the MST edges.
* **Result:** This distance is the **Pseudo-Time**. It represents the "severity score" or the evolutionary stage of the disease.

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
│   ├── raw/                # Original dataset 
│   ├── processed/          # Cleaned data
│   ├── results             # final CSV outputs (with patient trajectories) 
|
├── figures/                # Generated plots 
│
├── src/                    # Source Code Package
│   ├── __init__.py         # Package initialization
│   ├── preprocessing.py    # ETL, Imputation & Normalization
│   ├── euclidean_matrix.py # Distance Matrix Computation
│   ├── mst.py              # Graph Topology & Tree Construction
│   ├── trajectory.py       # Pathfinding (Dijkstra) & Sorting Logic
│   └── bootstrap.py        # Data Resampling Logic
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
    cd Pseudo-Time-Series-Thyroid-Diseases 
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

## 📚 Referências

1.  **Tucker, A., Li, Y., & Garway-Heath, D.** (2017). Updating Markov models to integrate cross-sectional and longitudinal studies. *Artificial Intelligence in Medicine*, 77, 23–30.
    * [DOI: 10.1016/j.artmed.2017.03.005](https://doi.org/10.1016/j.artmed.2017.03.005)

2.  **Puccio, B., Tucker, A., & Veltri, P.** (2024). Clustering Pseudo Time Series: Exploring Trajectories in the Ageing Process. In *pHealth 2024 Proceedings* (Studies in Health Technology and Informatics, Vol. 314, pp. 118–119). IOS Press.
    * [DOI: 10.3233/SHTI240070](https://doi.org/10.3233/SHTI240070)

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.