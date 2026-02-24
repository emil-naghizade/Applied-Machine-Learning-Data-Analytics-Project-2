# ===========================
# Train Energy Consumption Models
# ===========================

import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, PoissonRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    roc_curve, auc, r2_score, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

# ----------------------
# Load data
# ----------------------
train_df = pd.read_csv("train_energy_data.csv")
test_df  = pd.read_csv("test_energy_data.csv")

# ----------------------
# Preprocessing
# ----------------------
def preprocess(df, fit_encoders=None, fit_scalers=None):
    df_processed = df.copy()
    encoders = fit_encoders if fit_encoders else {}
    for col in ['Building Type', 'Day of Week']:
        if not fit_encoders:
            enc = LabelEncoder()
            df_processed[col] = enc.fit_transform(df_processed[col])
            encoders[col] = enc
        else:
            df_processed[col] = encoders[col].transform(df_processed[col])

    scalers = fit_scalers if fit_scalers else {}
    for col in ['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature']:
        if not fit_scalers:
            scaler = StandardScaler()
            df_processed[col] = scaler.fit_transform(df_processed[[col]])
            scalers[col] = scaler
        else:
            df_processed[col] = scalers[col].transform(df_processed[[col]])

    return df_processed, encoders, scalers

train_processed, encoders, scalers = preprocess(train_df)
test_processed, _, _               = preprocess(test_df, encoders, scalers)

features = [
    'Building Type', 'Square Footage', 'Number of Occupants',
    'Appliances Used', 'Average Temperature', 'Day of Week'
]

# ----------------------
# Targets
# ----------------------
median_energy = train_processed['Energy Consumption'].median()

train_processed['Energy Binary'] = (train_processed['Energy Consumption'] > median_energy).astype(int)
test_processed['Energy Binary']  = (test_processed['Energy Consumption']  > median_energy).astype(int)

# Tertile classes — use train quantiles for both splits
q_low  = train_processed['Energy Consumption'].quantile(1/3)
q_high = train_processed['Energy Consumption'].quantile(2/3)

def assign_class(series, q_low, q_high):
    return pd.cut(series, bins=[-np.inf, q_low, q_high, np.inf], labels=[0, 1, 2]).astype(int)

train_processed['Energy Class'] = assign_class(train_processed['Energy Consumption'], q_low, q_high)
test_processed['Energy Class']  = assign_class(test_processed['Energy Consumption'],  q_low, q_high)

# ----------------------
# 1a. Binary Logistic Regression
# ----------------------
lr_bin = LogisticRegression(max_iter=1000)
lr_bin.fit(train_processed[features], train_processed['Energy Binary'])

# 1b. Standard error, Z-statistic, p-value via statsmodels
X_sm         = sm.add_constant(train_processed[features].astype(float))
logit_model  = sm.Logit(train_processed['Energy Binary'], X_sm)
logit_result = logit_model.fit(disp=False)

logit_summary = pd.DataFrame({
    'Coefficient': logit_result.params,
    'Std_Error':   logit_result.bse,
    'Z_Statistic': logit_result.tvalues,
    'P_Value':     logit_result.pvalues
}).reset_index().rename(columns={'index': 'Variable'})

# 1c. Confounding analysis — correlation matrix + VIF
corr_matrix = train_processed[features + ['Energy Binary']].corr()

from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = train_processed[features].astype(float)
X_vif_const = sm.add_constant(X_vif)
vif_data = pd.DataFrame({
    'Feature': features,
    'VIF': [variance_inflation_factor(X_vif_const.values, i + 1) for i in range(len(features))]
})

# Flag potential confounders: corr with both another feature AND target > 0.3
confounder_flags = {}
for f in features:
    corr_with_target = abs(corr_matrix.loc[f, 'Energy Binary'])
    corr_with_others = corr_matrix[features].loc[f].drop(f).abs()
    is_confounder = corr_with_target > 0.3 and corr_with_others.max() > 0.3
    confounder_flags[f] = {
        'corr_with_target': round(corr_with_target, 3),
        'max_corr_with_others': round(corr_with_others.max(), 3),
        'potential_confounder': is_confounder
    }
confounder_df = pd.DataFrame(confounder_flags).T.reset_index().rename(columns={'index': 'Feature'})

# 1d. Multi-class Logistic Regression
lr_multi = LogisticRegression(multi_class='multinomial', max_iter=1000)
lr_multi.fit(train_processed[features], train_processed['Energy Class'])

# ----------------------
# 2a–d. Discriminant Analysis
# ----------------------
lda = LinearDiscriminantAnalysis()
lda.fit(train_processed[features], train_processed['Energy Binary'])

qda = QuadraticDiscriminantAnalysis()
qda.fit(train_processed[features], train_processed['Energy Binary'])

# ----------------------
# 3. Naive Bayes
# ----------------------
nb = GaussianNB()
nb.fit(train_processed[features], train_processed['Energy Binary'])

# ----------------------
# 4. Linear & Poisson Regression
# ----------------------
linreg = LinearRegression()
linreg.fit(train_processed[features], train_processed['Energy Consumption'])
linreg_preds = linreg.predict(test_processed[features])

poisreg = PoissonRegressor(max_iter=1000)
poisreg.fit(train_processed[features], train_processed['Energy Consumption'])
poisreg_preds = poisreg.predict(test_processed[features])

def regression_metrics(y_true, y_pred):
    return {
        'R2':   round(r2_score(y_true, y_pred), 4),
        'MSE':  round(mean_squared_error(y_true, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        'MAE':  round(mean_absolute_error(y_true, y_pred), 4),
    }

linreg_metrics  = regression_metrics(test_processed['Energy Consumption'], linreg_preds)
poisreg_metrics = regression_metrics(test_processed['Energy Consumption'], poisreg_preds)

regression_comparison = pd.DataFrame({
    'Metric': ['R²', 'MSE', 'RMSE', 'MAE'],
    'Linear Regression': [linreg_metrics['R2'], linreg_metrics['MSE'],
                          linreg_metrics['RMSE'], linreg_metrics['MAE']],
    'Poisson Regression': [poisreg_metrics['R2'], poisreg_metrics['MSE'],
                           poisreg_metrics['RMSE'], poisreg_metrics['MAE']],
})

# ----------------------
# ROC + optimal threshold
# ----------------------
def compute_roc(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, probs)
    optimal_idx = np.argmax(tpr - fpr)
    return {
        'fpr': fpr, 'tpr': tpr,
        'thresholds': thresholds,
        'optimal_threshold': float(thresholds[optimal_idx]),
        'auc': float(auc(fpr, tpr))
    }

roc_data = {}
for name, model in zip(['LR', 'LDA', 'QDA', 'NB'], [lr_bin, lda, qda, nb]):
    roc_data[name] = compute_roc(model, test_processed[features], test_processed['Energy Binary'])

# Multi-class OvR ROC for LR
ovr_lr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
ovr_lr.fit(train_processed[features], train_processed['Energy Class'])
roc_data['LR_multi'] = {}
for i, cls in enumerate([0, 1, 2]):
    probs = ovr_lr.predict_proba(test_processed[features])[:, i]
    y_bin = (test_processed['Energy Class'] == cls).astype(int)
    fpr, tpr, thresholds = roc_curve(y_bin, probs)
    roc_data['LR_multi'][cls] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': float(auc(fpr, tpr))}

# ----------------------
# Model comparison (binary classifiers)
# ----------------------
def classification_metrics(model, X, y, threshold=0.5):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        'Accuracy':  round(accuracy_score(y, preds), 4),
        'Precision': round(precision_score(y, preds, zero_division=0), 4),
        'Recall':    round(recall_score(y, preds, zero_division=0), 4),
        'F1':        round(f1_score(y, preds, zero_division=0), 4),
        'AUC':       round(roc_data[name]['auc'], 4) if name in roc_data else None
    }

comparison_rows = []
for name, model in zip(['LR', 'LDA', 'QDA', 'NB'], [lr_bin, lda, qda, nb]):
    opt_thresh = roc_data[name]['optimal_threshold']
    probs = model.predict_proba(test_processed[features])[:, 1]
    preds = (probs >= opt_thresh).astype(int)
    y_true = test_processed['Energy Binary']
    comparison_rows.append({
        'Model':     name,
        'Threshold': round(opt_thresh, 3),
        'Accuracy':  round(accuracy_score(y_true, preds), 4),
        'Precision': round(precision_score(y_true, preds, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, preds, zero_division=0), 4),
        'F1':        round(f1_score(y_true, preds, zero_division=0), 4),
        'AUC':       round(roc_data[name]['auc'], 4),
    })

model_comparison = pd.DataFrame(comparison_rows)

# ----------------------
# Feature importance
# ----------------------
lr_bin_importance  = pd.DataFrame({'Feature': features, 'Coefficient': lr_bin.coef_[0]})
linreg_importance  = pd.DataFrame({'Feature': features, 'Coefficient': linreg.coef_})

# LDA feature importance via scalings
lda_importance = pd.DataFrame({'Feature': features, 'Coefficient': lda.scalings_[:, 0]})

# ----------------------
# Save processed test data + all artifacts
# ----------------------
test_processed.to_csv("test_processed.csv", index=False)

with open("models.pkl", "wb") as f:
    pickle.dump({
        # Models
        'lr_bin':    lr_bin,
        'lr_multi':  lr_multi,
        'ovr_lr':    ovr_lr,
        'lda':       lda,
        'qda':       qda,
        'nb':        nb,
        'linreg':    linreg,
        'poisreg':   poisreg,
        # Encoders / scalers
        'encoders':  encoders,
        'scalers':   scalers,
        'features':  features,
        # Thresholds / targets
        'median_energy': median_energy,
        'q_low':  q_low,
        'q_high': q_high,
        # Analysis artifacts
        'logit_summary':       logit_summary,
        'corr_matrix':         corr_matrix,
        'vif_data':            vif_data,
        'confounder_df':       confounder_df,
        'roc_data':            roc_data,
        'model_comparison':    model_comparison,
        'regression_comparison': regression_comparison,
        'linreg_metrics':      linreg_metrics,
        'poisreg_metrics':     poisreg_metrics,
        'lr_bin_importance':   lr_bin_importance,
        'linreg_importance':   linreg_importance,
        'lda_importance':      lda_importance,
        # Raw predictions for residual plots
        'linreg_preds':  linreg_preds,
        'poisreg_preds': poisreg_preds,
        'y_test_reg':    test_processed['Energy Consumption'].values,
    }, f)

print("Training complete. All models and artifacts saved to models.pkl")