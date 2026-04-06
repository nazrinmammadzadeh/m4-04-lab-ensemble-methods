![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Ensemble Methods

## Overview

Individual models have their strengths, but combining multiple models — ensemble methods — often produces results that no single model can match on its own. This is why ensemble approaches like Random Forest, Gradient Boosting, and XGBoost dominate machine learning competitions and real-world applications alike.

In this lab, you'll work with the Wine Quality dataset to build a full ensemble pipeline: starting with individual baselines, moving through bagging and boosting, and finishing with stacking and voting classifiers. You'll see firsthand how combining weak learners creates a powerful predictive system.

The dataset presents a realistic challenge — wine quality is subjective and the classes are imbalanced — so you'll also practice handling the kind of messy classification problems you'll encounter in the real world.

## Learning Goals

By the end of this lab, you should be able to:

- Establish baselines with individual classifiers and compare them to ensemble approaches.
- Build and evaluate bagging and Random Forest models, interpreting feature importances.
- Train boosting models (AdaBoost, GradientBoosting) and analyze their learning curves.
- Combine models using VotingClassifier and StackingClassifier for improved performance.

## Setup and Context

You'll work inside a Jupyter Notebook for this lab. All analysis, code, and written interpretations should live in a single notebook so that your reasoning is visible alongside the output.

This lab builds on the ensemble methods lesson. You'll use scikit-learn's ensemble module extensively, along with pandas for data handling and matplotlib for visualization.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Getting Started

1. Create a new Jupyter Notebook called **`m4-04-ensemble-methods.ipynb`**.
2. Start with an import cell:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.metrics import accuracy_score, f1_score, classification_report

sns.set_style("whitegrid")
```

3. Work through the tasks in order. Each task builds on the previous one.
4. Include markdown cells between code cells to explain your observations and reasoning.

## Tasks

### Task 1: Baseline with Single Models

Load the data and establish individual model baselines.

1. Load the Wine Quality dataset and create a binary target:

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine = pd.read_csv(url, sep=";")
wine["quality_label"] = (wine["quality"] >= 7).astype(int)  # 1 = good, 0 = not good
```

2. Explore the dataset: how many features? What's the class distribution of `quality_label`? Is the dataset imbalanced?
3. Separate features and target. Drop the original `quality` column.
4. Split into training and test sets (80/20, `stratify=y`, `random_state=42`).
5. Scale features using `StandardScaler`.
6. Fit three baseline models:
   - `DecisionTreeClassifier(random_state=42)`
   - `LogisticRegression(max_iter=1000, random_state=42)`
   - `KNeighborsClassifier()`

7. Report **accuracy** and **F1 score** (use `f1_score` with `average="binary"`) for each on the test set. Organize results in a table.

### Task 2: Bagging & Random Forest

Explore how bagging improves on individual decision trees.

1. Fit a `BaggingClassifier` with a `DecisionTreeClassifier` as the base estimator (`n_estimators=100`, `oob_score=True`, `random_state=42`).
2. Fit a `RandomForestClassifier` (`n_estimators=100`, `oob_score=True`, `random_state=42`).
3. For both models, report:
   - Out-of-bag (OOB) score
   - Test accuracy and F1 score
4. Compare: How does bagging improve over the single Decision Tree from Task 1?
5. Plot the **top 10 feature importances** from the Random Forest as a horizontal bar chart. Which features matter most for predicting wine quality?
6. In a markdown cell, explain: Why does Random Forest typically outperform a single Decision Tree? What role does randomness play?

### Task 3: Boosting

Train boosting models and analyze how they learn.

1. Fit the following boosting models:
   - `AdaBoostClassifier(n_estimators=100, random_state=42)`
   - `GradientBoostingClassifier(n_estimators=100, random_state=42)`
   - (Optional) `HistGradientBoostingClassifier(max_iter=100, random_state=42)` — scikit-learn's faster implementation

2. Report accuracy and F1 for each on the test set.

3. **Learning curves:** For `GradientBoostingClassifier`, use `staged_predict` to compute training and test accuracy at each boosting stage (1 to 100 estimators). Plot both curves on the same figure. At what point does the model start to overfit (if at all)?

4. In a markdown cell, compare AdaBoost and GradientBoosting: How do they differ in their approach? When might you prefer one over the other?

### Task 4: Stacking & Voting

Combine your best models into meta-ensembles.

1. **Voting classifier:** Select the 3 best-performing models from Tasks 1–3. Build a `VotingClassifier` with `voting="soft"`. Report accuracy and F1 on the test set.

2. **Stacking classifier:** Using the same 3 base models, build a `StackingClassifier` with `LogisticRegression()` as the `final_estimator`. Report accuracy and F1 on the test set.

3. **Final comparison table:** Create a comprehensive DataFrame comparing **all** models from the entire lab (baselines, bagging, Random Forest, boosting, voting, stacking) with their accuracy and F1 scores. Sort by F1 descending.

4. In a concluding markdown cell, answer:
   - Which ensemble strategy performed best on this dataset?
   - Was the improvement over single models significant?
   - What are the trade-offs (training time, interpretability, complexity) of using ensemble methods?
   - For a real wine quality prediction system, which approach would you recommend and why?

## Submission

### What to submit

- `m4-04-ensemble-methods.ipynb` — your completed notebook with all code, outputs, and markdown explanations.

### Definition of done (checklist)

- [ ] Wine Quality dataset is loaded with binary target and explored for class balance.
- [ ] Three baseline models are trained and evaluated.
- [ ] BaggingClassifier and RandomForestClassifier are trained, with OOB scores and feature importance plot.
- [ ] AdaBoost and GradientBoosting are trained, with a learning curve plot for GradientBoosting.
- [ ] VotingClassifier and StackingClassifier combine the best models.
- [ ] A final comparison table includes all models with accuracy and F1 scores.
- [ ] Markdown cells explain ensemble concepts and justify model recommendations.
- [ ] The notebook runs top-to-bottom without errors (`Kernel → Restart & Run All`).

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete ensemble methods comparison"
git push origin main
```

Then open a **Pull Request** on the original repository with a brief description of your work.
