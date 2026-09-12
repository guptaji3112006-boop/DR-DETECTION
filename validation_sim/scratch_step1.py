import pandas as pd
import json

audit_file = r"validation_sim\results\baseline\20260912_111803_170834\prediction_audit.csv"
df = pd.read_csv(audit_file)

# Only consider successful inference
df = df[df['inference_status'] == 'SUCCESS']

# True Positives: true_grade >= 2 AND predicted_grade >= 2
# True Negatives: true_grade < 2 AND predicted_grade < 2
# False Positives: true_grade < 2 AND predicted_grade >= 2
# False Negatives: true_grade >= 2 AND predicted_grade < 2

tp = len(df[(df['true_grade'] >= 2) & (df['predicted_grade'] >= 2)])
tn = len(df[(df['true_grade'] < 2) & (df['predicted_grade'] < 2)])
fp = len(df[(df['true_grade'] < 2) & (df['predicted_grade'] >= 2)])
fn = len(df[(df['true_grade'] >= 2) & (df['predicted_grade'] < 2)])

print(f"Total: {len(df)}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Sensitivity: {tp/(tp+fn):.2%}")
print(f"Specificity: {tn/(tn+fp):.2%}")

# Let's inspect False Positives by true grade and predicted grade
fp_df = df[(df['true_grade'] < 2) & (df['predicted_grade'] >= 2)]
print("\nFalse Positives by True Grade:")
print(fp_df['true_grade'].value_counts().sort_index())

print("\nFalse Positives by Predicted Grade:")
print(fp_df['predicted_grade'].value_counts().sort_index())

print("\nOverall Confusion Matrix (True x Predicted):")
cm = pd.crosstab(df['true_grade'], df['predicted_grade'], margins=True)
print(cm)

print("\nPer-Class Recall:")
for c in range(5):
    true_c = len(df[df['true_grade'] == c])
    if true_c > 0:
        correct_c = len(df[(df['true_grade'] == c) & (df['predicted_grade'] == c)])
        print(f"Class {c}: {correct_c}/{true_c} ({correct_c/true_c:.2%})")
