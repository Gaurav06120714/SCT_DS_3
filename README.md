# Bank Term Deposit Prediction — Decision Tree Classifier

**SkillCraft Technology | Data Science Internship | Task 03**

---

> *Can a 5-minute phone call predict whether someone opens a bank account?*  
> This project answers that question using machine learning on real-world banking data.

---

## What I Built

A **Decision Tree Classifier** trained on 41,000+ real bank marketing call records to predict whether a customer will subscribe to a term deposit — based on who they are, how they were contacted, and what the economy looked like at the time.

---

## Repository Structure

```
SCT_DS_3/
│
├── SCT_DS_3.py                  ← Python script (run this)
├── SCT_DS_3.ipynb               ← Same thing, Jupyter notebook
│
├── bank-additional-full.csv     ← Raw dataset (41,188 rows)
├── bank_cleaned.csv             ← After label encoding (auto-generated)
│
├── images/
│   ├── class_distribution.png
│   ├── age_distribution.png
│   ├── correlation_heatmap.png
│   ├── depth_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importances.png
│   ├── feature_importances_horizontal.png
│   └── decision_tree.png
│
└── README.md
```

---

## Dataset at a Glance

| | |
|---|---|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **Rows** | 41,188 customer call records |
| **Features** | 20 (age, job, marital status, call duration, economic indicators…) |
| **Target** | `y` — Did the customer subscribe? (`yes` / `no`) |
| **Class split** | 88.7% No · 11.3% Yes ← heavily imbalanced |

---

## My Approach

### 1 · Exploratory Data Analysis
Before touching the model, I explored the data to understand what I was working with:
- The dataset is **heavily imbalanced** — only 1 in 9 customers said yes
- `"unknown"` is used as a placeholder in several categorical columns (not null)
- Call `duration` shows a strong visual separation between yes/no groups even before modeling

### 2 · Preprocessing
- Label-encoded all 11 categorical columns (`job`, `marital`, `education`, `contact`, etc.)
- Target `y` → 0 (No) / 1 (Yes)
- Exported the encoded version as `bank_cleaned.csv` for reproducibility

### 3 · Finding the Right Depth
Instead of just picking `max_depth=5` arbitrarily, I trained the model at every depth from 3 to 10 and plotted both train and test accuracy:

> Train accuracy keeps climbing → that's overfitting.  
> Test accuracy peaks around depth 5 and flattens → that's the sweet spot.

### 4 · Training & Evaluation
- `DecisionTreeClassifier(max_depth=5, criterion='gini')`
- 80/20 stratified train-test split
- Evaluated with accuracy, F1-score, confusion matrix, and ROC-AUC

---

## Results

| Metric | Score |
|---|---|
| Train Accuracy | **91.63%** |
| Test Accuracy | **91.81%** |
| AUC Score | **0.9335** |
| F1 (Yes class) | 0.59 |

> The 91% accuracy sounds great but is misleading — because 89% of the data is "No", a model that always predicts No would also get 89%. The **F1-score for "Yes"** (0.59) and **AUC (0.93)** are the real indicators of model quality here.

---

## Key Findings

**#1 — Call duration dominates everything**  
`duration` alone accounts for ~50% of the model's decision weight. Longer calls = higher chance of yes. But there's a catch: you only know the duration *after* the call ends, so this feature isn't useful for pre-call predictions.

**#2 — The economy matters more than demographics**  
`nr.employed` (number of employees in the economy) and `euribor3m` (interest rate) rank 2nd and 3rd. When interest rates drop and employment is high, people are more likely to open savings products.

**#3 — Who you call matters less than when you call**  
Age, job, and marital status have surprisingly low importance. Timing (economic climate + previous campaign outcome) is far more predictive than customer demographics alone.

---

## Visualizations

| Chart | What it shows |
|---|---|
| `class_distribution.png` | The 89/11 imbalance in the target |
| `age_distribution.png` | Spread of customer ages |
| `correlation_heatmap.png` | Which features move together |
| `depth_comparison.png` | Why I chose depth=5 |
| `confusion_matrix.png` | Where the model gets it right and wrong |
| `roc_curve.png` | AUC = 0.9335 |
| `feature_importances.png` | Top 10 drivers of prediction |
| `feature_importances_horizontal.png` | Same, horizontal layout |
| `decision_tree.png` | The actual tree (depth 3 for readability) |

---

## How to Run

```bash
# Clone and enter the folder
git clone https://github.com/Gaurav06120714/SCT_DS_3.git
cd SCT_DS_3

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the script — all PNGs and bank_cleaned.csv will be generated
python SCT_DS_3.py

# Or open the notebook
jupyter notebook SCT_DS_3.ipynb
```

---

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn`

---

## Author

**Gaurav** — Data Science Intern @ SkillCraft Technology  
GitHub: [@Gaurav06120714](https://github.com/Gaurav06120714)
