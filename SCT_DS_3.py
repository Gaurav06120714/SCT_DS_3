# =============================================================
# SCT_DS_3 — Decision Tree Classifier | Bank Marketing Dataset
# SkillCraft Technology Internship — Task 03
# =============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

# Output folder for all images
IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

# ── 1. Load Dataset ───────────────────────────────────────────
df = pd.read_csv("bank-additional-full.csv", sep=";")

print("=" * 55)
print("STEP 1 — Dataset Loaded")
print("=" * 55)
print(f"Shape       : {df.shape}")
print(f"Columns     : {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# ── 2. EDA ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2 — EDA")
print("=" * 55)
print(f"\nTarget distribution:\n{df['y'].value_counts()}")
print(f"\n% breakdown:\n{df['y'].value_counts(normalize=True).mul(100).round(2)}")

# Class imbalance bar chart
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["y"].value_counts()
ax.bar(counts.index, counts.values, color=["#5C85D6", "#EF5350"], edgecolor="white", width=0.5)
ax.set_title("Target Class Distribution", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Subscribed to Term Deposit")
ax.set_ylabel("Count")
for i, (label, val) in enumerate(counts.items()):
    ax.text(i, val + 200, f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=9)
ax.set_xticks([0, 1])
ax.set_xticklabels(["No", "Yes"])
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: class_distribution.png")

# Age distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["age"], bins=40, color="#5C85D6", edgecolor="white", alpha=0.85)
ax.set_title("Age Distribution of Clients", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/age_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: age_distribution.png")

# ── 3. Preprocessing ──────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3 — Preprocessing")
print("=" * 55)

label_cols = df.select_dtypes(include="object").columns.tolist()
le = LabelEncoder()
df_encoded = df.copy()
for col in label_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

df_encoded.to_csv("bank_cleaned.csv", index=False)
print(f"Encoded {len(label_cols)} categorical columns: {label_cols}")
print("Saved: bank_cleaned.csv")

# Correlation heatmap (numeric only)
fig, ax = plt.subplots(figsize=(14, 10))
corr = df_encoded.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", linewidths=0.3,
            ax=ax, vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: correlation_heatmap.png")

# ── 4. Train / Test Split ─────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4 — Train / Test Split (80 / 20, stratified)")
print("=" * 55)

X = df_encoded.drop(columns=["y"])
y = df_encoded["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set : {X_train.shape[0]:,} rows")
print(f"Test set     : {X_test.shape[0]:,} rows")

# ── 5. Depth Comparison ───────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5 — Optimal Depth Search (depth 3–10)")
print("=" * 55)

depths = range(3, 11)
train_scores, test_scores = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, criterion="gini", random_state=42)
    m.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, m.predict(X_train)))
    test_scores.append(accuracy_score(y_test, m.predict(X_test)))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(depths, [s * 100 for s in train_scores], "o-", label="Train Accuracy", color="#5C85D6")
ax.plot(depths, [s * 100 for s in test_scores], "s--", label="Test Accuracy", color="#EF5350")
ax.axvline(x=5, color="gray", linestyle=":", linewidth=1.2, label="Selected depth=5")
ax.set_title("Accuracy vs Tree Depth", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Max Depth")
ax.set_ylabel("Accuracy (%)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/depth_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: depth_comparison.png")

# ── 6. Train Final Model ──────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6 — Training Final Model (max_depth=5, gini)")
print("=" * 55)

clf = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc  = accuracy_score(y_test, y_pred)
print(f"Train Accuracy : {train_acc * 100:.2f}%")
print(f"Test Accuracy  : {test_acc  * 100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['No','Yes'])}")

# ── 7. Confusion Matrix ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
            linewidths=0.5, linecolor="white", annot_kws={"size": 14})
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: confusion_matrix.png")

# ── 8. ROC Curve ─────────────────────────────────────────────
y_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#5C85D6", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.2, label="Random Classifier")
ax.set_title("ROC Curve", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: roc_curve.png  |  AUC = {roc_auc:.4f}")

# ── 9. Feature Importances ────────────────────────────────────
fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Vertical bar chart (top 10)
fig, ax = plt.subplots(figsize=(10, 5))
fi.head(10).plot(kind="bar", ax=ax, color="#5C85D6", edgecolor="white")
ax.set_title("Top 10 Feature Importances", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Feature")
ax.set_ylabel("Importance Score")
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/feature_importances.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: feature_importances.png")

# Horizontal bar chart
fig, ax = plt.subplots(figsize=(8, 6))
fi.sort_values(ascending=True).tail(10).plot(kind="barh", ax=ax, color="#5C85D6", edgecolor="white")
ax.set_title("Top 10 Feature Importances (Horizontal)", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/feature_importances_horizontal.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: feature_importances_horizontal.png")

# ── 10. Decision Tree Visualization ──────────────────────────
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(clf, feature_names=X.columns.tolist(), class_names=["No", "Yes"],
          filled=True, rounded=True, fontsize=7, ax=ax, max_depth=3)
plt.title("Decision Tree Visualization (Depth 3)", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/decision_tree.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: decision_tree.png")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("RESULTS SUMMARY")
print("=" * 55)
print(f"  Train Accuracy : {train_acc * 100:.2f}%")
print(f"  Test Accuracy  : {test_acc  * 100:.2f}%")
print(f"  AUC Score      : {roc_auc:.4f}")
print(f"\n  Top 3 Features:")
for feat, score in fi.head(3).items():
    print(f"    {feat:<25} {score:.4f}")
print("\nAll outputs saved. Done.")
