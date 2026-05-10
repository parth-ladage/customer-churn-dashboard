"""
Customer Churn Prediction & Risk Segmentation Dashboard
========================================================
Built with Streamlit, Plotly, Scikit-learn, XGBoost.
Author: Parth Ladage
"""

import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS (read as UTF-8 to handle Unicode characters)
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Plotly dark-theme helper
# ──────────────────────────────────────────────────────────────
def styled_fig(fig):
    """Apply consistent dark styling to every Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter"),
        title_font=dict(color="#e2e8f0"),
        legend=dict(font=dict(color="#cbd5e1")),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="rgba(99,102,241,0.12)", zerolinecolor="rgba(99,102,241,0.12)")
    fig.update_yaxes(gridcolor="rgba(99,102,241,0.12)", zerolinecolor="rgba(99,102,241,0.12)")
    return fig


# ──────────────────────────────────────────────────────────────
# Data loading & preprocessing
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(csv_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df["Churn_Binary"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


# ──────────────────────────────────────────────────────────────
# Model training (cached)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(_df):
    df = _df.copy()

    # Encode categoricals
    le_dict = {}
    cat_cols = df.select_dtypes(include="object").columns.drop(["customerID", "Churn"])
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop(columns=["customerID", "Churn", "Churn_Binary"])
    y = df["Churn_Binary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42, verbosity=0),
    }

    results = {}
    roc_data = {}
    feature_importances = {}

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
        }

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr)

        if hasattr(model, "feature_importances_"):
            feature_importances[name] = dict(zip(X.columns, model.feature_importances_))

    # Determine best model
    best_model_name = max(results, key=lambda k: results[k]["ROC-AUC"])
    best_model = models[best_model_name]
    best_auc = results[best_model_name]["ROC-AUC"]

    # Score entire dataset for risk segmentation
    X_all_scaled = scaler.transform(X)
    all_probs = best_model.predict_proba(X_all_scaled)[:, 1]

    return results, roc_data, feature_importances, all_probs, X.columns.tolist(), best_model_name, best_auc


# ──────────────────────────────────────────────────────────────
# Load data & train models
# ──────────────────────────────────────────────────────────────
raw_df = load_raw_data()

# ──────────────────────────────────────────────────────────────
# Sidebar filters
# ──────────────────────────────────────────────────────────────
st.sidebar.markdown(
    '## <i class="fa-solid fa-sliders"></i> Filters',
    unsafe_allow_html=True,
)

contract_options = sorted(raw_df["Contract"].unique())
selected_contracts = st.sidebar.multiselect("Contract Type", contract_options, default=contract_options)

internet_options = sorted(raw_df["InternetService"].unique())
selected_internet = st.sidebar.multiselect("Internet Service", internet_options, default=internet_options)

payment_options = sorted(raw_df["PaymentMethod"].unique())
selected_payment = st.sidebar.multiselect("Payment Method", payment_options, default=payment_options)

senior_options = {0: "No", 1: "Yes"}
selected_senior = st.sidebar.multiselect("Senior Citizen", list(senior_options.values()), default=list(senior_options.values()))
selected_senior_vals = [k for k, v in senior_options.items() if v in selected_senior]

tenure_range = st.sidebar.slider(
    "Tenure Range (months)",
    int(raw_df["tenure"].min()),
    int(raw_df["tenure"].max()),
    (int(raw_df["tenure"].min()), int(raw_df["tenure"].max())),
)

# Apply filters
filtered_df = raw_df[
    (raw_df["Contract"].isin(selected_contracts))
    & (raw_df["InternetService"].isin(selected_internet))
    & (raw_df["PaymentMethod"].isin(selected_payment))
    & (raw_df["SeniorCitizen"].isin(selected_senior_vals))
    & (raw_df["tenure"].between(tenure_range[0], tenure_range[1]))
].copy()

if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust your selections.")
    st.stop()

# Train models on full data (cached)
results, roc_data, feature_importances, all_probs, feature_names, best_model_name, best_auc = train_models(raw_df)

# Attach risk scores to filtered view
prob_series = pd.Series(all_probs, index=raw_df.index)
filtered_df["Churn_Prob"] = prob_series.reindex(filtered_df.index)
filtered_df["Risk_Tier"] = pd.cut(
    filtered_df["Churn_Prob"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"],
)


# ══════════════════════════════════════════════════════════════
#  1.  HEADER  &  KPIs
# ══════════════════════════════════════════════════════════════
st.markdown(
    '<p class="dashboard-title">'
    '<i class="fa-solid fa-chart-column"></i> Customer Churn Prediction & Risk Segmentation'
    "</p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="dashboard-subtitle">'
    "Predicting customer churn and identifying high-risk customers using machine learning"
    "</p>",
    unsafe_allow_html=True,
)

total = len(filtered_df)
churn_rate = filtered_df["Churn_Binary"].mean() * 100
high_risk_pct = (filtered_df["Risk_Tier"] == "High Risk").mean() * 100

k1, k2, k3, k4 = st.columns(4)
for col, label, value in [
    (k1, "TOTAL CUSTOMERS", f"{total:,}"),
    (k2, "CHURN RATE", f"{churn_rate:.1f}%"),
    (k3, "HIGH RISK CUSTOMERS", f"{high_risk_pct:.1f}%"),
    (k4, "BEST MODEL ROC-AUC", f"{best_auc:.4f}"),
]:
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
#  2.  CUSTOMER  CHURN  ANALYSIS
# ══════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">'
    '<i class="fa-solid fa-magnifying-glass-chart"></i> Customer Churn Analysis'
    "</p>",
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

with c1:
    churn_counts = filtered_df["Churn"].value_counts()
    fig_pie = px.pie(
        values=churn_counts.values,
        names=churn_counts.index,
        title="Churn Distribution",
        color_discrete_sequence=["#22c55e", "#ef4444"],
        hole=0.45,
    )
    fig_pie.update_traces(textinfo="percent+label", textfont_size=13)
    st.plotly_chart(styled_fig(fig_pie), use_container_width=True)
    st.markdown(
        '<div class="insight-box">'
        '<i class="fa-solid fa-lightbulb" style="color:#facc15;"></i> '
        "<b>Insight:</b> Approximately 1 in 4 customers churns, "
        "signaling a significant retention challenge for the business."
        "</div>",
        unsafe_allow_html=True,
    )

with c2:
    churn_by_contract = filtered_df.groupby("Contract")["Churn_Binary"].mean().reset_index()
    churn_by_contract["Churn_Binary"] *= 100
    churn_by_contract.columns = ["Contract", "Churn Rate"]
    fig_bar = px.bar(
        churn_by_contract,
        x="Contract",
        y="Churn Rate",
        color="Contract",
        text=churn_by_contract["Churn Rate"].round(1),
        title="Churn Rate by Contract Type",
        color_discrete_sequence=["#6366f1", "#a78bfa", "#c084fc"],
    )
    fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_bar.update_yaxes(title_text="Churn Rate (%)")
    st.plotly_chart(styled_fig(fig_bar), use_container_width=True)
    st.markdown(
        '<div class="insight-box">'
        '<i class="fa-solid fa-lightbulb" style="color:#facc15;"></i> '
        "<b>Insight:</b> Month-to-month customers churn at ~42%, "
        "far exceeding one-year (~11%) and two-year (~3%) contract holders."
        "</div>",
        unsafe_allow_html=True,
    )

# Row 2 — Tenure distribution & scatter
c3, c4 = st.columns(2)

with c3:
    fig_kde = go.Figure()
    for label, color in [("No", "#22c55e"), ("Yes", "#ef4444")]:
        subset = filtered_df[filtered_df["Churn"] == label]["tenure"]
        if len(subset) > 2:
            try:
                kde = gaussian_kde(subset)
                x_vals = np.linspace(0, 72, 300)
                fig_kde.add_trace(go.Scatter(x=x_vals, y=kde(x_vals), mode="lines", name=label, line=dict(color=color, width=2), fill="tozeroy"))
            except Exception:
                pass
    fig_kde.update_layout(title="Tenure Distribution: Churned vs Retained", xaxis_title="Tenure (months)", yaxis_title="Density")
    st.plotly_chart(styled_fig(fig_kde), use_container_width=True)
    st.markdown(
        '<div class="insight-box">'
        '<i class="fa-solid fa-lightbulb" style="color:#facc15;"></i> '
        "<b>Insight:</b> Churned customers have an average tenure of ~18 months vs ~38 months for retained. "
        "The first 6-12 months are critical for retention."
        "</div>",
        unsafe_allow_html=True,
    )

with c4:
    sample = filtered_df.sample(min(800, len(filtered_df)), random_state=42)
    fig_scatter = px.scatter(
        sample,
        x="tenure",
        y="MonthlyCharges",
        color="Churn",
        color_discrete_map={"Yes": "#ef4444", "No": "#22c55e"},
        opacity=0.6,
        title="Tenure vs Monthly Charges",
    )
    fig_scatter.update_traces(marker=dict(size=5))
    fig_scatter.update_xaxes(title_text="Tenure (months)")
    fig_scatter.update_yaxes(title_text="Monthly Charges ($)")
    st.plotly_chart(styled_fig(fig_scatter), use_container_width=True)
    st.markdown(
        '<div class="insight-box">'
        '<i class="fa-solid fa-lightbulb" style="color:#facc15;"></i> '
        "<b>Insight:</b> Churned customers tend to have higher monthly charges and shorter tenure, "
        "suggesting pricing pressure drives early exits."
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
#  3.  MODEL  PERFORMANCE
# ══════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">'
    '<i class="fa-solid fa-trophy" style="color:#facc15;"></i> Model Performance'
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    f'<span class="best-model-badge">'
    f'<i class="fa-solid fa-star"></i> Best Model: {best_model_name} (ROC-AUC {best_auc:.4f})'
    f"</span>",
    unsafe_allow_html=True,
)

results_df = pd.DataFrame(results).T.sort_values(by="ROC-AUC", ascending=False)
st.dataframe(results_df.style.format("{:.4f}").highlight_max(axis=0, color="#312e81"), use_container_width=True)

# ROC curves
fig_roc = go.Figure()
colors_roc = ["#6366f1", "#8b5cf6", "#a855f7", "#f59e0b"]
for idx, (name, (fpr, tpr)) in enumerate(roc_data.items()):
    fig_roc.add_trace(
        go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{name} (AUC={results[name]['ROC-AUC']:.4f})",
            line=dict(color=colors_roc[idx % len(colors_roc)], width=2),
        )
    )
fig_roc.add_trace(
    go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(dash="dash", color="#4b5563"),
    )
)
fig_roc.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(styled_fig(fig_roc), use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  4.  FEATURE  IMPORTANCE
# ══════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">'
    '<i class="fa-solid fa-thumbtack" style="color:#ef4444;"></i> Feature Importance (Top 10)'
    "</p>",
    unsafe_allow_html=True,
)

# Choose best tree-based model for feature importance
imp_model_name = None
for candidate in ["XGBoost", "Gradient Boosting", "Random Forest"]:
    if candidate in feature_importances:
        imp_model_name = candidate
        break

if imp_model_name:
    imp = feature_importances[imp_model_name]
    imp_df = pd.DataFrame({"Feature": list(imp.keys()), "Importance": list(imp.values())})
    imp_df = imp_df.sort_values("Importance", ascending=True).tail(10)

    fig_imp = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        title=f"Top 10 Features -- {imp_model_name}",
        color="Importance",
        color_continuous_scale=["#312e81", "#6366f1", "#a855f7"],
    )
    fig_imp.update_coloraxes(showscale=False)
    st.plotly_chart(styled_fig(fig_imp), use_container_width=True)
    st.markdown(
        '<div class="insight-box">'
        '<i class="fa-solid fa-lightbulb" style="color:#facc15;"></i> '
        "<b>Key Drivers:</b><br>"
        "- <b>Tenure</b> -- Longer tenure strongly reduces churn risk.<br>"
        "- <b>Contract Type</b> -- Month-to-month contracts are the highest churn predictor.<br>"
        "- <b>Monthly Charges</b> -- Higher charges correlate with increased churn.<br>"
        "- <b>Fiber Optic Service</b> -- Fiber optic users show elevated churn, possibly due to pricing or service quality."
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
#  5.  RISK  SEGMENTATION
# ══════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">'
    '<i class="fa-solid fa-triangle-exclamation" style="color:#f59e0b;"></i> Risk Segmentation'
    "</p>",
    unsafe_allow_html=True,
)

r1, r2 = st.columns([1, 2])

with r1:
    risk_counts = filtered_df["Risk_Tier"].value_counts()
    fig_risk = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Customer Risk Distribution",
        color_discrete_sequence=["#22c55e", "#f59e0b", "#ef4444"],
        hole=0.5,
    )
    fig_risk.update_traces(textinfo="percent+label", textfont_size=12)
    st.plotly_chart(styled_fig(fig_risk), use_container_width=True)

with r2:
    rc1, rc2, rc3 = st.columns(3)
    for col, tier, css_class, icon_html in [
        (rc1, "High Risk", "risk-card-high", '<i class="fa-solid fa-circle" style="color:#ef4444;"></i>'),
        (rc2, "Medium Risk", "risk-card-medium", '<i class="fa-solid fa-circle" style="color:#f59e0b;"></i>'),
        (rc3, "Low Risk", "risk-card-low", '<i class="fa-solid fa-circle" style="color:#22c55e;"></i>'),
    ]:
        subset = filtered_df[filtered_df["Risk_Tier"] == tier]
        avg_tenure = subset["tenure"].mean() if len(subset) > 0 else 0
        avg_monthly = subset["MonthlyCharges"].mean() if len(subset) > 0 else 0
        count = len(subset)
        top_contract = subset["Contract"].mode().iloc[0] if len(subset) > 0 else "N/A"
        col.markdown(
            f"""
            <div class="{css_class}">
                <h3 style="margin:0;color:white;">{icon_html} {tier}</h3>
                <p style="color:#d1d5db;margin:0.6rem 0 0.3rem;">Customers: <b>{count}</b></p>
                <p style="color:#d1d5db;margin:0.3rem 0;">Avg Tenure: <b>{avg_tenure:.1f} mo</b></p>
                <p style="color:#d1d5db;margin:0.3rem 0;">Avg Monthly: <b>${avg_monthly:.2f}</b></p>
                <p style="color:#d1d5db;margin:0.3rem 0;">Top Contract: <b>{top_contract}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="insight-box">'
    '<i class="fa-solid fa-lightbulb" style="color:#facc15;"></i> '
    "<b>Recommendation:</b> Medium-risk customers should be targeted first for retention campaigns -- "
    "they represent the highest ROI opportunity as they are still persuadable."
    "</div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
#  6.  BUSINESS  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">'
    '<i class="fa-solid fa-clipboard-list"></i> Business Recommendations'
    "</p>",
    unsafe_allow_html=True,
)

recommendations = [
    (
        '<i class="fa-solid fa-rocket" style="color:#a855f7;"></i> Improve Onboarding',
        "Focus on the first 6-12 months. Implement welcome programs, proactive check-ins, and onboarding guides to build early loyalty.",
    ),
    (
        '<i class="fa-solid fa-file-contract" style="color:#6366f1;"></i> Promote Long-Term Contracts',
        "Offer discounts or perks for customers who commit to one-year or two-year contracts. This dramatically reduces churn.",
    ),
    (
        '<i class="fa-solid fa-coins" style="color:#f59e0b;"></i> Optimize Pricing Strategies',
        "High monthly charges drive churn. Consider tiered pricing, loyalty discounts, or bundling to retain price-sensitive customers.",
    ),
    (
        '<i class="fa-solid fa-headset" style="color:#22c55e;"></i> Better Support for Premium Users',
        "Fiber optic and premium service users churn more. Invest in dedicated support channels and faster issue resolution.",
    ),
]

rec_cols = st.columns(2)
for i, (title, desc) in enumerate(recommendations):
    rec_cols[i % 2].markdown(
        f'<div class="rec-card"><h3>{title}</h3><p>{desc}</p></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
#  7.  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown(
    '<div class="footer-text">'
    "Built with <b>Streamlit</b> &middot; Plotly &middot; Scikit-learn &middot; XGBoost &middot; Pandas<br>"
    'Author: <b>Parth Ladage</b> &middot; <a href="https://github.com/ParthLadworker" target="_blank">GitHub</a>'
    " &middot; "
    '<a href="https://linkedin.com/in/parth-ladage" target="_blank">LinkedIn</a>'
    "</div>",
    unsafe_allow_html=True,
)
