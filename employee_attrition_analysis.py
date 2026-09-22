import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(df.head())

print("\n=== Dataset Shape ===")
print(df.shape)

print("\n=== Column Names ===")
print(df.columns.tolist())

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Duplicate Rows ===")
print(df.duplicated().sum())

print("\n=== Attrition Count ===")
print(df["Attrition"].value_counts())

# Visualization 1: Attrition Rate by Overtime

overtime_attrition = (
    df.groupby("OverTime")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

print("\n=== Attrition Rate by Overtime ===")
print(overtime_attrition)

plt.figure(figsize=(7, 5))

overtime_attrition.plot(kind="bar")

plt.title("Attrition Rate by Overtime")
plt.xlabel("OverTime")
plt.ylabel("Attrition Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# Visualization 2: Monthly Income vs Attrition

plt.figure(figsize=(8, 5))

df.boxplot(
    column="MonthlyIncome",
    by="Attrition"
)

plt.title("Monthly Income by Attrition")
plt.suptitle("")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")

plt.tight_layout()
plt.show()

# Visualization 3: Job Satisfaction vs Attrition

plt.figure(figsize=(8, 5))

attrition_job_satisfaction = pd.crosstab(
    df["JobSatisfaction"],
    df["Attrition"],
    normalize="index"
) * 100

attrition_job_satisfaction.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Attrition Rate by Job Satisfaction")
plt.xlabel("Job Satisfaction")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.legend(title="Attrition")
plt.tight_layout()
plt.show()

# Visualization 4: Attrition by Job Role

plt.figure(figsize=(10, 5))

attrition_job_role = pd.crosstab(
    df["JobRole"],
    df["Attrition"],
    normalize="index"
) * 100

attrition_job_role.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Attrition Rate by Job Role")
plt.xlabel("Job Role")
plt.ylabel("Attrition Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Attrition")
plt.tight_layout()
plt.show()

# Visualization 5: Attrition by Age

plt.figure(figsize=(8, 5))

age_attrition = pd.crosstab(
    df["Age"],
    df["Attrition"],
    normalize="index"
) * 100

age_attrition["Yes"].plot(kind="line", figsize=(8, 5))

plt.title("Attrition Rate by Age")
plt.xlabel("Age")
plt.ylabel("Attrition Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualization 6: Attrition by Job Involvement

plt.figure(figsize=(8, 5))

job_involvement_attrition = pd.crosstab(
    df["JobInvolvement"],
    df["Attrition"],
    normalize="index"
) * 100

job_involvement_attrition.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Attrition Rate by Job Involvement")
plt.xlabel("Job Involvement")
plt.ylabel("Attrition Rate (%)")
plt.xticks(rotation=0)
plt.legend(title="Attrition")
plt.tight_layout()
plt.show()

# ============================================================
# OBSERVATIONS
# ============================================================

print("\n=== Key Observations ===")

print("1. Employees working overtime have a higher attrition rate than employees who do not work overtime.")

print("2. Employees who left the company generally have lower monthly income than employees who stayed.")

print("3. Employees with lower job satisfaction show higher attrition compared with employees having higher job satisfaction.")

print("4. Attrition varies considerably across different job roles, with Sales Representatives showing relatively high attrition.")

print("5. Younger employees show higher attrition rates compared with many older age groups.")

print("6. Employees with lower job involvement have higher attrition compared with employees having higher job involvement.")



# ============================================================
# MACHINE LEARNING - EMPLOYEE ATTRITION PREDICTION
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

print("\n=== MACHINE LEARNING ===")

# Create a copy of the dataset
ml_df = df.copy()

# Encode target variable
ml_df["Attrition"] = ml_df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# Convert categorical variables into numerical dummy variables
ml_df = pd.get_dummies(ml_df, drop_first=True)

# Separate features and target
columns_to_drop = [
    "Attrition",
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours"
]

X = ml_df.drop(columns=columns_to_drop)
y = ml_df["Attrition"]

# Split BEFORE SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nOriginal training distribution:")
print(y_train.value_counts())

# ============================================================
# SMOTE - BALANCE TRAINING DATA ONLY
# ============================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# ============================================================
# LOGISTIC REGRESSION
# ============================================================

log_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

log_model.fit(X_train_smote, y_train_smote)

y_pred_log = log_model.predict(X_test)

print("\n=== Logistic Regression + SMOTE ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# ============================================================
# RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train_smote, y_train_smote)

y_pred_rf = rf_model.predict(X_test)

print("\n=== Random Forest + SMOTE ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ============================================================
# CONFUSION MATRIX - LOGISTIC REGRESSION + SMOTE
# ============================================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm_smote = confusion_matrix(y_test, y_pred_log)

print("\n=== Confusion Matrix - Logistic Regression + SMOTE ===")
print(cm_smote)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_smote,
    display_labels=["No Attrition", "Attrition"]
)

disp.plot()

plt.title("Confusion Matrix - Logistic Regression + SMOTE")
plt.tight_layout()
plt.show()

# ============================================================
# CONFUSION MATRIX - RANDOM FOREST + SMOTE
# ============================================================

cm_rf_smote = confusion_matrix(y_test, y_pred_rf)

print("\n=== Confusion Matrix - Random Forest + SMOTE ===")
print(cm_rf_smote)

disp_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf_smote,
    display_labels=["No Attrition", "Attrition"]
)

disp_rf.plot()

plt.title("Confusion Matrix - Random Forest + SMOTE")
plt.tight_layout()
plt.show()

# ============================================================
# ROC-AUC COMPARISON
# ============================================================

from sklearn.metrics import roc_auc_score, RocCurveDisplay

# Logistic Regression probabilities
y_prob_log = log_model.predict_proba(X_test)[:, 1]

# Random Forest probabilities
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC
auc_log = roc_auc_score(y_test, y_prob_log)
auc_rf = roc_auc_score(y_test, y_prob_rf)

print("\n=== ROC-AUC Comparison ===")
print("Logistic Regression + SMOTE:", round(auc_log, 4))
print("Random Forest + SMOTE:", round(auc_rf, 4))

# ROC curves
plt.figure(figsize=(8, 6))

RocCurveDisplay.from_predictions(
    y_test,
    y_prob_log,
    name="Logistic Regression + SMOTE"
)

RocCurveDisplay.from_predictions(
    y_test,
    y_prob_rf,
    name="Random Forest + SMOTE"
)

plt.title("ROC Curve Comparison")
plt.tight_layout()
plt.show()

# ============================================================
# LOGISTIC REGRESSION - FEATURE COEFFICIENTS
# ============================================================

feature_coefficients = pd.Series(
    log_model.coef_[0],
    index=X.columns
).sort_values()

print("\n=== Top 10 Factors Associated With Higher Attrition Prediction ===")
print(feature_coefficients.tail(10).sort_values(ascending=False))

print("\n=== Top 10 Factors Associated With Lower Attrition Prediction ===")
print(feature_coefficients.head(10))


y_prob_log = log_model.predict_proba(X_test)[:, 1]

# ============================================================
# EMPLOYEE ATTRITION RISK LEVEL
# ============================================================

# Create a risk table from the test data
risk_df = df.loc[X_test.index].copy()

# Add predicted attrition probability
risk_df["Attrition_Probability"] = y_prob_log

# Assign risk level
def assign_risk(probability):
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    else:
        return "High"

risk_df["Risk_Level"] = risk_df["Attrition_Probability"].apply(assign_risk)

# Sort employees from highest to lowest risk
risk_df = risk_df.sort_values(
    by="Attrition_Probability",
    ascending=False
)

# Select top 20 employees
top_20_risk = risk_df[
    ["EmployeeNumber", "Attrition_Probability", "Risk_Level"]
].head(20)

print("\n=== TOP 20 EMPLOYEE ATTRITION RISK ===")
print(top_20_risk.to_string(index=False))


# ============================================================
# OBSERVATIONS VS POSSIBLE INSIGHTS
# ============================================================

print("\n=== OBSERVATIONS VS POSSIBLE INSIGHTS ===")

print("\n1. Observation:")
print("Employees working overtime show a higher attrition rate than employees who do not work overtime.")
print("Possible Insight:")
print("Overtime is associated with higher employee attrition risk, suggesting that workload patterns may be relevant for retention monitoring.")

print("\n2. Observation:")
print("Employees who left the company generally have lower monthly income than employees who stayed.")
print("Possible Insight:")
print("Monthly income shows an association with attrition patterns, indicating that compensation may be relevant when examining employee retention.")

print("\n3. Observation:")
print("Employees with lower job satisfaction show higher attrition compared with employees having higher job satisfaction.")
print("Possible Insight:")
print("Job satisfaction is associated with differences in attrition patterns and may be useful when identifying employees requiring retention attention.")

print("\n4. Observation:")
print("Attrition varies across different job roles, with some roles showing higher attrition than others.")
print("Possible Insight:")
print("Job role is associated with different levels of attrition risk, suggesting that retention strategies may need to consider role-specific patterns.")

print("\n5. Observation:")
print("Younger employees show higher attrition rates compared with many older age groups.")
print("Possible Insight:")
print("Age groups show different attrition patterns, with younger employees appearing more frequently in higher-attrition groups.")

print("\n6. Observation:")
print("Employees with lower job involvement have higher attrition compared with employees having higher job involvement.")
print("Possible Insight:")
print("Job involvement is associated with attrition patterns and may be relevant when assessing employee retention risk.")

# ============================================================
# BUSINESS ACTIONS
# ============================================================

print("\n=== 5 BUSINESS ACTIONS ===")

print("""
1. Overtime Management:
Monitor employees working overtime and review workload distribution
to reduce excessive workload and potential attrition risk.

2. Employee Satisfaction Programs:
Conduct regular employee satisfaction surveys and address concerns
related to job satisfaction and work environment.

3. Compensation and Career Growth:
Review compensation, promotion opportunities, and career development
for employees showing higher attrition risk.

4. Role-Specific Retention:
Identify job roles with higher attrition patterns and develop
role-specific retention and engagement strategies.

5. Early Risk Monitoring:
Use the ML-based attrition probability and Risk_Level classification
to identify employees requiring retention attention and prioritize
appropriate HR interventions.
""")