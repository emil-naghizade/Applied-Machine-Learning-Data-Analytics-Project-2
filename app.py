# ===========================
# Energy Consumption — Streamlit App
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# ─────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("models.pkl", "rb") as f:
        return pickle.load(f)

data = load_models()

lr_bin          = data['lr_bin']
lr_multi        = data['lr_multi']
ovr_lr          = data['ovr_lr']
lda             = data['lda']
qda             = data['qda']
nb              = data['nb']
linreg          = data['linreg']
poisreg         = data['poisreg']
encoders        = data['encoders']
scalers         = data['scalers']
features        = data['features']
median_energy   = data['median_energy']
q_low           = data['q_low']
q_high          = data['q_high']
logit_summary   = data['logit_summary']
corr_matrix     = data['corr_matrix']
vif_data        = data['vif_data']
confounder_df   = data['confounder_df']
roc_data        = data['roc_data']
model_comparison    = data['model_comparison']
regression_comparison = data['regression_comparison']
linreg_metrics  = data['linreg_metrics']
poisreg_metrics = data['poisreg_metrics']
lr_bin_importance  = data['lr_bin_importance']
linreg_importance  = data['linreg_importance']
lda_importance     = data['lda_importance']
linreg_preds    = data['linreg_preds']
poisreg_preds   = data['poisreg_preds']
y_test_reg      = data['y_test_reg']

# Load saved test set (avoids re-running training)
@st.cache_data
def load_test():
    return pd.read_csv("test_processed.csv")

test_processed = load_test()

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def preprocess_input(df):
    df_p = df.copy()
    for col in ['Building Type', 'Day of Week']:
        df_p[col] = encoders[col].transform(df_p[col])
    for col in ['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature']:
        df_p[col] = scalers[col].transform(df_p[[col]])
    return df_p

def styled_metric(label, value, delta=None):
    st.metric(label=label, value=value, delta=delta)

COLOR_MAP = {'LR': '#4C72B0', 'LDA': '#DD8452', 'QDA': '#55A868', 'NB': '#C44E52'}

# ─────────────────────────────────────────
# Sidebar — Input Features
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
    st.title("⚡ Input Features")
    st.markdown("---")

    building_type = st.selectbox("🏢 Building Type", encoders['Building Type'].classes_)
    day_of_week   = st.selectbox("📅 Day of Week",   encoders['Day of Week'].classes_)
    sq_footage    = st.number_input("📐 Square Footage (sq ft)", min_value=100.0,    max_value=50000.0, value=2000.0,  step=100.0)
    n_occupants   = st.number_input("👥 Number of Occupants",    min_value=1.0,      max_value=500.0,   value=10.0,   step=1.0)
    appliances    = st.number_input("🔌 Appliances Used",        min_value=0.0,      max_value=100.0,   value=5.0,    step=1.0)
    avg_temp      = st.number_input("🌡️ Average Temperature (°C)", min_value=-30.0,  max_value=60.0,    value=22.0,   step=0.5)

    st.markdown("---")
    run_predict = st.button("🔮 Run Prediction", use_container_width=True)

user_df = pd.DataFrame([{
    'Building Type': building_type,
    'Square Footage': sq_footage,
    'Number of Occupants': n_occupants,
    'Appliances Used': appliances,
    'Average Temperature': avg_temp,
    'Day of Week': day_of_week,
}])
user_df_processed = preprocess_input(user_df)

# ─────────────────────────────────────────
# Main — Tabs
# ─────────────────────────────────────────
st.title("⚡ Energy Consumption Prediction App")
st.markdown("Apply ML models to predict building energy consumption — select a tab to explore.")

tabs = st.tabs([
    "🔮 Predictions",
    "📊 Logistic Regression",
    "📈 Discriminant Analysis",
    "🤖 Model Comparison",
    "📉 Regression Models",
    "🔍 Confounding Analysis",
    "💡 Feature Importance",
])

# ═══════════════════════════════════════════
# TAB 1 — PREDICTIONS
# ═══════════════════════════════════════════
with tabs[0]:
    st.header("🔮 Model Predictions")

    if run_predict:
        st.success("Predictions generated for your inputs!")

        col1, col2 = st.columns(2)

        # ── Binary classifiers ──
        with col1:
            st.subheader("Binary Classifiers  (High / Low energy)")
            for name, model in zip(['LR', 'LDA', 'QDA', 'NB'], [lr_bin, lda, qda, nb]):
                opt_thresh = roc_data[name]['optimal_threshold']
                prob = model.predict_proba(user_df_processed[features])[0]
                pred = int(prob[1] >= opt_thresh)
                label = "🔴 HIGH" if pred == 1 else "🟢 LOW"
                with st.expander(f"{name}  →  {label}  (p={prob[1]:.3f})", expanded=True):
                    threshold = st.slider(
                        f"Decision threshold — {name}",
                        0.0, 1.0,
                        float(opt_thresh),
                        key=f"thresh_{name}"
                    )
                    pred2 = int(prob[1] >= threshold)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Prediction", "HIGH" if pred2 == 1 else "LOW")
                    c2.metric("P(High)", f"{prob[1]:.3f}")
                    c3.metric("AUC", f"{roc_data[name]['auc']:.3f}")

        # ── Multi-class + regression ──
        with col2:
            st.subheader("Multi-class & Regression")

            with st.expander("Logistic Regression — 3 Classes", expanded=True):
                prob_mc = lr_multi.predict_proba(user_df_processed[features])[0]
                pred_mc = lr_multi.predict(user_df_processed[features])[0]
                class_labels = {0: "Low", 1: "Medium", 2: "High"}
                st.metric("Predicted Class", class_labels[pred_mc])
                fig, ax = plt.subplots(figsize=(4, 2.5))
                ax.bar(class_labels.values(), prob_mc, color=['#55A868', '#4C72B0', '#C44E52'])
                ax.set_ylabel("Probability")
                ax.set_title("Class Probabilities")
                st.pyplot(fig)
                plt.close()

            with st.expander("Linear Regression — Continuous Prediction", expanded=True):
                lin_pred = linreg.predict(user_df_processed[features])[0]
                st.metric("Predicted Energy (kWh)", f"{lin_pred:.2f}")
                st.caption(f"Test R² = {linreg_metrics['R2']}")

            with st.expander("Poisson Regression — Continuous Prediction", expanded=True):
                pois_pred = poisreg.predict(user_df_processed[features])[0]
                st.metric("Predicted Energy (kWh)", f"{pois_pred:.2f}")
                st.caption(f"Test R² = {poisreg_metrics['R2']}")

    else:
        st.info("👈 Set your features in the sidebar and click **Run Prediction** to see results.")

# ═══════════════════════════════════════════
# TAB 2 — LOGISTIC REGRESSION
# ═══════════════════════════════════════════
with tabs[1]:
    st.header("📊 Logistic Regression Details")

    st.subheader("1b. Coefficient Table — SE, Z-Statistic, p-Value")
    st.caption("Computed via statsmodels Logit.")

    def highlight_pval(val):
        if isinstance(val, float):
            if val < 0.001:  return 'background-color: #c6efce; color: #276221'
            if val < 0.05:   return 'background-color: #ffeb9c; color: #9c6500'
            return 'background-color: #ffc7ce; color: #9c0006'
        return ''

    display_logit = logit_summary.copy()
    for col in ['Coefficient', 'Std_Error', 'Z_Statistic', 'P_Value']:
        display_logit[col] = display_logit[col].round(4)

    st.dataframe(
        display_logit.style.applymap(highlight_pval, subset=['P_Value']),
        use_container_width=True
    )
    st.caption("🟢 p < 0.001  |  🟡 p < 0.05  |  🔴 p ≥ 0.05")

    st.markdown("---")
    st.subheader("1d. Multi-class ROC Curves (One-vs-Rest)")
    fig, ax = plt.subplots(figsize=(7, 4))
    class_colors = ['#4C72B0', '#DD8452', '#55A868']
    class_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    for i, cls in enumerate([0, 1, 2]):
        rd = roc_data['LR_multi'][cls]
        ax.plot(rd['fpr'], rd['tpr'], color=class_colors[i],
                label=f"Class {class_labels[cls]} (AUC={rd['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-class LR — OvR ROC Curves")
    ax.legend()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════
# TAB 3 — DISCRIMINANT ANALYSIS
# ═══════════════════════════════════════════
with tabs[2]:
    st.header("📈 Discriminant Analysis")

    st.subheader("2b–c. ROC Curves with Optimal Thresholds")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, name, model, title in zip(
        axes,
        ['LDA', 'QDA'],
        [lda, qda],
        ['Linear Discriminant Analysis (LDA)', 'Quadratic Discriminant Analysis (QDA)']
    ):
        rd = roc_data[name]
        ax.plot(rd['fpr'], rd['tpr'], color=COLOR_MAP[name], linewidth=2,
                label=f"AUC = {rd['auc']:.3f}")
        # Mark optimal point
        opt_idx = np.argmax(rd['tpr'] - rd['fpr'])
        ax.scatter(rd['fpr'][opt_idx], rd['tpr'][opt_idx], color='red', zorder=5,
                   label=f"Optimal threshold = {rd['optimal_threshold']:.3f}")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend()

    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("LDA Optimal Threshold", f"{roc_data['LDA']['optimal_threshold']:.3f}")
        st.metric("LDA AUC",               f"{roc_data['LDA']['auc']:.3f}")
    with col2:
        st.metric("QDA Optimal Threshold", f"{roc_data['QDA']['optimal_threshold']:.3f}")
        st.metric("QDA AUC",               f"{roc_data['QDA']['auc']:.3f}")

# ═══════════════════════════════════════════
# TAB 4 — MODEL COMPARISON (LR, LDA, QDA, NB)
# ═══════════════════════════════════════════
with tabs[3]:
    st.header("🤖 Model Comparison — LR · LDA · QDA · NB")

    st.subheader("Performance Metrics at Optimal Threshold")
    styled = model_comparison.style \
        .highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
                       color='#c6efce') \
        .format(precision=4)
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")
    st.subheader("AUC Comparison — ROC Curves")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, model in zip(['LR', 'LDA', 'QDA', 'NB'], [lr_bin, lda, qda, nb]):
        rd = roc_data[name]
        ax.plot(rd['fpr'], rd['tpr'], color=COLOR_MAP[name], linewidth=2,
                label=f"{name} (AUC={rd['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Binary Classifiers")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Metric Bar Chart")
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(metrics_to_plot))
    width = 0.18
    for i, (_, row) in enumerate(model_comparison.iterrows()):
        ax.bar(x + i * width, [row[m] for m in metrics_to_plot],
               width=width, label=row['Model'], color=list(COLOR_MAP.values())[i])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════
# TAB 5 — REGRESSION MODELS
# ═══════════════════════════════════════════
with tabs[4]:
    st.header("📉 Linear vs Poisson Regression")

    st.subheader("4. Performance Metrics")
    st.dataframe(
        regression_comparison.style.highlight_max(
            subset=['Linear Regression', 'Poisson Regression'], axis=1, color='#c6efce'
        ).format(precision=4),
        use_container_width=True
    )
    st.caption("🟢 Green = better value for each metric (higher R², lower MSE/RMSE/MAE).")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test_reg, linreg_preds,  alpha=0.4, s=10, label='Linear',  color='#4C72B0')
        ax.scatter(y_test_reg, poisreg_preds, alpha=0.4, s=10, label='Poisson', color='#DD8452')
        mn, mx = y_test_reg.min(), y_test_reg.max()
        ax.plot([mn, mx], [mn, mx], 'k--', linewidth=0.8, label='Ideal')
        ax.set_xlabel("Actual Energy (kWh)")
        ax.set_ylabel("Predicted Energy (kWh)")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Residuals")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(linreg_preds,  y_test_reg - linreg_preds,
                   alpha=0.4, s=10, label='Linear Residuals',  color='#4C72B0')
        ax.scatter(poisreg_preds, y_test_reg - poisreg_preds,
                   alpha=0.4, s=10, label='Poisson Residuals', color='#DD8452')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel("Predicted Energy (kWh)")
        ax.set_ylabel("Residual")
        ax.set_title("Residual Plot")
        ax.legend()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════
# TAB 6 — CONFOUNDING ANALYSIS
# ═══════════════════════════════════════════
with tabs[5]:
    st.header("🔍 Confounding Variable Analysis")

    st.subheader("1c. Potential Confounders")
    st.caption("A feature is flagged if it correlates strongly with both the target (|r| > 0.3) and another feature (|r| > 0.3).")
    conf_display = confounder_df.copy()
    conf_display['potential_confounder'] = conf_display['potential_confounder'].map({True: '⚠️ Yes', False: '✅ No'})
    st.dataframe(conf_display, use_container_width=True)

    st.markdown("---")
    st.subheader("Variance Inflation Factor (VIF)")
    st.caption("VIF > 5 suggests multicollinearity; VIF > 10 is severe.")
    vif_display = vif_data.copy()
    vif_display['VIF'] = vif_display['VIF'].round(3)

    def color_vif(val):
        if isinstance(val, float):
            if val > 10: return 'background-color: #ffc7ce'
            if val > 5:  return 'background-color: #ffeb9c'
            return 'background-color: #c6efce'
        return ''

    st.dataframe(vif_display.style.applymap(color_vif, subset=['VIF']), use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Matrix Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
        center=0, linewidths=0.5, ax=ax
    )
    ax.set_title("Feature & Target Correlation Matrix")
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════
# TAB 7 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════
with tabs[6]:
    st.header("💡 Feature Importance")

    col1, col2, col3 = st.columns(3)

    def importance_chart(ax, df, title, color):
        sorted_df = df.reindex(df['Coefficient'].abs().sort_values().index)
        colors = [color if c >= 0 else '#C44E52' for c in sorted_df['Coefficient']]
        ax.barh(sorted_df['Feature'], sorted_df['Coefficient'], color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Coefficient")

    with col1:
        st.subheader("Binary Logistic Regression")
        fig, ax = plt.subplots(figsize=(5, 4))
        importance_chart(ax, lr_bin_importance, "LR Binary Coefficients", '#4C72B0')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Linear Regression")
        fig, ax = plt.subplots(figsize=(5, 4))
        importance_chart(ax, linreg_importance, "Linear Reg Coefficients", '#55A868')
        st.pyplot(fig)
        plt.close()

    with col3:
        st.subheader("LDA Scalings")
        fig, ax = plt.subplots(figsize=(5, 4))
        importance_chart(ax, lda_importance, "LDA Feature Scalings", '#DD8452')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Raw Coefficient Tables")
    t1, t2, t3 = st.tabs(["LR Binary", "Linear Regression", "LDA"])
    with t1:
        st.dataframe(lr_bin_importance.sort_values('Coefficient', key=abs, ascending=False), use_container_width=True)
    with t2:
        st.dataframe(linreg_importance.sort_values('Coefficient', key=abs, ascending=False), use_container_width=True)
    with t3:
        st.dataframe(lda_importance.sort_values('Coefficient', key=abs, ascending=False), use_container_width=True)