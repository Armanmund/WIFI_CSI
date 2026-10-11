# Project Objectives: Cross-Environment Adaptation for Wi-Fi CSI Activity Recognition

**Project Title:** Cross-Environment Adaptation for Wi-Fi CSI-Based Human Activity Recognition  
**Degree / Context:** B.Tech Final Year Project  
**Benchmark Dataset:** ReWiS Public Dataset (CSI matrices for `walk`, `stand`, `jump`, `empty` across environments `A1`, `A2`, `A3`)  
**Model Architecture:** `CSIActivityCNN` (4-block 2D CNN with adaptive pooling, ~785k parameters)

---

## Core Objectives

1. **Establish a Source-Environment Baseline (Experiment 1):**  
   Train and evaluate a compact 2D Convolutional Neural Network on the source environment (**A1**) using ReWiS Channel State Information (CSI) to determine the upper-bound benchmark accuracy across four activity classes (*walking*, *standing*, *jumping*, and *empty room*).

2. **Quantify Environmental Domain Shift (Experiment 2):**  
   Perform zero-shot evaluation of the A1-trained baseline model on unseen target rooms (**A2** and **A3**) without any retraining to measure the exact accuracy penalty caused by room-specific multipath RF distortion.

3. **Measure Few-Shot Adaptation Cost (Experiment 3):**  
   Fine-tune the pre-trained model with deliberately restricted sample budgets per class ($k \in \{1, 2, 3, 5, 10, 15, 20, 30, 50\}$ shots) across multiple random seeds and sample draws to quantify the exact volume of new labeled data required to restore baseline performance.

4. **Isolate Pre-Training Advantage (Experiment 4 - Control):**  
   Train control models from random weight initialization (from scratch) using identical few-shot sample subsets to scientifically isolate and prove the performance gain provided by source pre-training versus naive target training.

5. **Derive the Characteristic Recovery Curve (Deliverable):**  
   Generate statistically validated accuracy-versus-samples curves with 95% confidence intervals across multiple random seeds and sample draws, establishing an empirical deployment cost metric and sample count threshold (e.g., 90% recovery) for deploying Wi-Fi CSI activity recognition in unseen environments.

---

## Experimental Protocol & Deliverables

| Stage | Objective Addressed | Input Data | Target Output |
| :--- | :--- | :--- | :--- |
| **Exp 1: Baseline** | Objective 1 | `train_A1` (85/15 train/val split) | Benchmark accuracy, `baseline_A1.pt`, training loss/accuracy curves |
| **Exp 2: Cross-Env** | Objective 2 | `test_A2`, `test_A3` (Zero-shot) | Transfer accuracy, accuracy drop metric, confusion matrices |
| **Exp 3: Fine-Tuning** | Objective 3 | $k$ shots/class from `test_A2`, `test_A3` | Target accuracy vs. $k$ shots curve, 95% CI |
| **Exp 4: From Scratch**| Objective 4 | Same $k$ shots/class (Random weights) | Control curve isolating transfer learning margin |
| **Synthesis & Plots** | Objective 5 | All aggregated trials (25 runs / $k$) | `finetuning_curve_A2.png`, `recovery_curve_A2.png`, `summary_results.csv` |
