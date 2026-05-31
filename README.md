# Task 03 — Decision Tree Classifier
### SkillCraft Technology Internship

---

## Objective

Build a decision tree classifier to predict whether a customer will subscribe to a bank term deposit, based on demographic and behavioral data from the UCI Bank Marketing Dataset.

---

## Dataset

**Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

**File used:** `bank-additional-full.csv`

| Property | Value |
|---|---|
| Total rows | 41,188 |
| Total features | 20 |
| Target column | `y` (yes / no) |
| Separator | semicolon (`;`) |
| Missing values | None (uses `"unknown"` as placeholder) |

---

## Project Structure

```
SCT_DS_3/
│
├── bank-additional-full.csv           # Raw dataset
├── bank_cleaned.csv                   # Encoded dataset (generated on run)
│
├── SCT_DS_3.py                        # Main Python script
├── SCT_DS_3.ipynb                     # Jupyter Notebook (same steps, interactive)
│
├── images/
│   ├── class_distribution.png        # Target class bar chart
│   ├── age_distribution.png          # Age histogram
│   ├── correlation_heatmap.png       # Feature correlation heatmap
│   ├── depth_comparison.png          # Train vs test accuracy by depth
│   ├── confusion_matrix.png          # Confusion matrix heatmap
│   ├── roc_curve.png                 # ROC curve with AUC
│   ├── feature_importances.png       # Top 10 features (vertical)
│   ├── feature_importances_horizontal.png  # Top 10 features (horizontal)
│   └── decision_tree.png             # Visualized decision tree (depth 3)
│
└── README.md                         # This file
```

---

## Steps Followed

### Step 1 — Load Dataset
Loaded `bank-additional-full.csv` using `pandas.read_csv()` with `sep=';'`.  
Verified shape, column types, and confirmed zero null values.

### Step 2 — EDA
- Analyzed target class distribution → heavily imbalanced (~89% No, ~11% Yes)
- Visualized age distribution of all clients
- Checked `"unknown"` string placeholders in categorical columns

### Step 3 — Preprocessing
- Applied `LabelEncoder` to all categorical (`object`) columns
- Encoded target `y` → 0 (No), 1 (Yes)
- Generated correlation heatmap across all features
- Saved encoded dataset as `bank_cleaned.csv`

### Step 4 — Train / Test Split
- 80% training / 20% test split
- Used `stratify=y` to preserve the class imbalance ratio in both sets
- Training set: ~32,950 rows | Test set: ~8,238 rows

### Step 5 — Optimal Depth Search
- Trained models at depths 3–10
- Plotted train vs test accuracy to find the best depth
- Selected `max_depth=5` — best balance between accuracy and overfitting

### Step 6 — Model Training
- Trained `DecisionTreeClassifier` with `max_depth=5`, `criterion='gini'`
- Evaluated on held-out test set

### Step 7 — Evaluation
- Accuracy, precision, recall, F1-score via `classification_report`
- Confusion matrix heatmap
- ROC curve + AUC score

### Step 8 — Visualization
- Full decision tree plotted using `plot_tree()` (limited to depth 3 for readability)
- Feature importances as both vertical and horizontal bar charts

---

## Results

| Metric | Value |
|---|---|
| Training Accuracy | ~91% |
| Test Accuracy | ~90% |
| AUC Score | ~0.80 |

---

## Key Findings

- **`duration`** (last contact duration in seconds) is the single most important feature — longer calls strongly indicate that the customer agreed to subscribe.
- **Economic indicators** (`euribor3m`, `nr.employed`, `emp.var.rate`) are the next most influential, showing that macroeconomic conditions drive term deposit uptake.
- The dataset is **highly imbalanced** (9:1 ratio of No to Yes). Accuracy alone is misleading — F1-score and recall for the "Yes" class are the meaningful metrics.
- Beyond `max_depth=5`, training accuracy climbs while test accuracy plateaus — a textbook sign of overfitting in decision trees.
- The model achieves reasonable AUC (~0.80), meaning it has solid discriminative power despite the class imbalance.

---

## Libraries Used

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install all with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run

**Option A — Python script:**
```bash
# Place bank-additional-full.csv in the same folder, then:
python SCT_DS_3.py
```

**Option B — Jupyter Notebook:**
```bash
jupyter notebook SCT_DS_3.ipynb
# Run all cells top to bottom
```

All output PNGs and `bank_cleaned.csv` are generated automatically on run.

---

## Author

**Intern:** Gaurav  
**Organization:** SkillCraft Technology  
**Task:** Task 03 — Decision Tree Classifier
# SCT_DS_3
