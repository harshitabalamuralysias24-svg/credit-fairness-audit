# credit-fairness-audit
Fairness audit of a logistic regression credit-risk model on the German Credit dataset, with bias mitigation and trade-off analysis
# Fairness Audit of a Credit-Risk Model

## Question
Does a logistic regression credit-risk model treat applicants differently by
age or sex, and what does it cost to reduce those differences?

## Data
German Credit dataset (OpenML, 1,000 applicants, ~30% labelled bad credit risk).
Sensitive attributes: age (under 25 vs 25 and over) and sex (derived from the
personal_status field). 70/30 stratified train/test split.

## Methods
1. Baseline logistic regression (scikit-learn): accuracy 0.783, recall 0.533, ROC-AUC 0.801.
2. Fairness audit with fairlearn: selection rate, false positive rate and false
   negative rate by group, plus demographic parity and equalized odds differences.
3. Mitigation: fairlearn ThresholdOptimizer (equalized odds constraint).
4. Proxy check: retrained the model without age and personal_status.

## Key results
- Baseline: applicants under 25 had a false
