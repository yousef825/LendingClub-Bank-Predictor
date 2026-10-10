
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

st.set_page_config(page_title="LendingClub Loan Default Dashboard", layout="wide")
st.title("LendingClub Loan Default Dashboard")

RANDOM_STATE = 42
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
GRADE_MAP = {g: i for i, g in enumerate(GRADE_ORDER)}
HOME_OWNERSHIP_OPTS = ["RENT", "MORTGAGE", "OWN", "OTHER"]
PURPOSE_OPTS = [
    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
    "small_business", "car", "medical", "moving", "vacation", "house",
    "wedding", "renewable_energy", "other",
]
EMP_LENGTH_OPTS = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}
NUMERIC_BASE_COLS = [
    "loan_amnt", "term", "int_rate", "installment", "emp_length",
    "annual_inc", "dti", "fico_range_low", "open_acc", "pub_rec", "revol_util",
]
FEATURE_COLS_BASE = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "fico_range_low", "open_acc", "pub_rec", "revol_util", "emp_length", "term",
    "payment_to_income", "loan_to_income", "total_interest", "has_pub_rec", "grade_num",
]
CATEGORICAL_TO_ENCODE = ["home_ownership", "purpose"]


# Load the cleaned Lending Club dataset with Features Engineering
file_path = 'D:/LendingCLub/Datasets/sample_of_accepted_loans_clean_with_features.csv'
df_feat_full = pd.read_csv(file_path, low_memory=False)

# Convert the issue date to datetime if it exists.
if "issue_d" in df_feat_full.columns:
    df_feat_full["issue_d"] = pd.to_datetime(
        df_feat_full["issue_d"],
        errors="coerce"
    )


# REQUIRED COLUMNS

required_columns = {
    "loan_status",
    "target",
    "grade",
    "loan_amnt",
    "term",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "purpose",
    "home_ownership",
    "revol_util",
    "open_acc",
    "pub_rec",
}

missing_columns = required_columns - set(df_feat_full.columns)

if missing_columns:
    st.error(
        f"The dataset is missing these required columns:\n"
        f"{sorted(missing_columns)}"
    )
    st.stop()


def clean_and_engineer(df):
    ...
    return df


# BUILD FEATURE MATRIX

def build_feature_matrix(df, ref_dummy_cols=None):

    df = df.copy()

    # Numerical features used by the models
    numeric_features = [
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "fico_range_low",
        "open_acc",
        "pub_rec",
        "revol_util",
        "payment_to_income",
        "loan_to_income",
        "total_interest",
        "grade_num",
        "has_pub_rec",
    ]

    numeric_features = [
        col for col in numeric_features
        if col in df.columns
    ]

    # Categorical features
    categorical_features = [
        "grade",
        "purpose",
        "home_ownership",
    ]

    categorical_features = [
        col for col in categorical_features
        if col in df.columns
    ]

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(
        df,
        columns=categorical_features,
        drop_first=True,
        dtype=int,
    )

    # Save or reuse dummy columns
    if ref_dummy_cols is None:

        dummy_cols = [
            col
            for col in df_encoded.columns
            if any(
                col.startswith(f"{cat}_")
                for cat in categorical_features
            )
        ]

    else:

        dummy_cols = ref_dummy_cols

        # Add missing dummy columns
        for col in dummy_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

    # Create final feature list
    feature_cols = numeric_features + dummy_cols

    # Keep only existing columns
    feature_cols = [
        col
        for col in feature_cols
        if col in df_encoded.columns
    ]

    return df_encoded, feature_cols, dummy_cols    




# MODEL TRAINING

@st.cache_resource(show_spinner="Training machine learning models...")
def train_models(df: pd.DataFrame):

    # Convert categorical columns into numeric features.
    df_encoded, feature_cols, dummy_cols = build_feature_matrix(df)

    # Split the data into features (X) and target (y).
    X = df_encoded[feature_cols]
    y = df_encoded["target"]

    # Split the dataset into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Standardize the numeric features.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Balance the training data using SMOTE
    if HAS_SMOTE:
        smote = SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=5
        )

        X_train_final, y_train_final = smote.fit_resample(
            X_train_scaled,
            y_train
        )

    else:
        X_train_final = X_train_scaled
        y_train_final = y_train

    # Machine Learning Models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    # Add XGBoost if it is installed.
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )

    # Train and Evaluate Models
    results = {}

    for model_name, model in models.items():

        model.fit(
            X_train_final,
            y_train_final
        )

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        results[model_name] = {

            "model": model,

            "Accuracy": accuracy_score(y_test, y_pred),

            "Precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "F1": f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "ROC-AUC": roc_auc_score(
                y_test,
                y_prob
            ),

            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    # Select the model with the highest F1 Score.
    best_name = max(
        results,
        key=lambda name: results[name]["F1"]
    )

    return {
        "results": results,
        "best_name": best_name,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "dummy_cols": dummy_cols,
    }

# SINGLE LOAN PREDICTION
# Predict whether a new loan is likely to default.

def predict_single(inputs: dict, bundle: dict, model_name: str):

    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    dummy_cols = bundle["dummy_cols"]

    model = bundle["results"][model_name]["model"]

    # Convert the user's input into a DataFrame.
    loan = pd.DataFrame([inputs])

    # Create the engineered features used during training.
    loan["payment_to_income"] = (
        loan["installment"] /
        (loan["annual_inc"] / 12 + 1)
    )

    loan["loan_to_income"] = (
        loan["loan_amnt"] /
        (loan["annual_inc"] + 1)
    )

    loan["total_interest"] = (
        loan["installment"] * loan["term"]
    ) - loan["loan_amnt"]

    loan["has_pub_rec"] = (
        loan["pub_rec"] > 0
    ).astype(int)

    loan["grade_num"] = loan["grade"].map(GRADE_MAP)

    # Apply the same encoding used during training.
    loan_encoded, _, _ = build_feature_matrix(
        loan,
        ref_dummy_cols=dummy_cols
    )

    X = loan_encoded[feature_cols]

    X_scaled = scaler.transform(X)

    # Predict the probability of default
    probability = model.predict_proba(X_scaled)[0, 1]

    prediction = int(probability >= 0.50)

    return prediction, probability


# SIDEBAR — FILTERS
st.sidebar.header("Filters")


# LOAN GRADE FILTER
# Display only the loan grades that exist in the dataset 
available_grades = [
    grade
    for grade in GRADE_ORDER
    if grade in df_feat_full["grade"].dropna().unique()
]

selected_grades = st.sidebar.multiselect(
    label="Loan Grade",
    options=available_grades,
    default=available_grades,
)


# LOAN PURPOSE FILTER
# Display all available loan purposes

available_purposes = sorted(df_feat_full["purpose"].dropna().unique())

selected_purposes = st.sidebar.multiselect(
    label="Loan Purpose",
    options=["All Purposes"] + list(available_purposes),
    default=["All Purposes"],
)

# If "All Purposes" is selected, use every purpose.
if "All Purposes" in selected_purposes:
    selected_purposes = list(available_purposes)

# LOAN TERM FILTER
# Display the available loan terms.

available_terms = sorted(
    df_feat_full["term"]
    .dropna()
    .unique()
)

selected_terms = st.sidebar.multiselect(
    label="Loan Term (Months)",
    options=available_terms,
    default=available_terms,
)



# LOAN STATUS FILTER
# Display the available loan status values.

available_status = sorted(df_feat_full["loan_status"].dropna().astype(str).unique())

selected_status = st.sidebar.multiselect(
    label="Loan Status",
    options=available_status,
    default=available_status,
)

# APPLY FILTERS
# Keep only the loans that match the selected filters.

filtered_df = df_feat_full[
    (df_feat_full["grade"].isin(selected_grades))
    & (df_feat_full["purpose"].isin(selected_purposes))
    & (df_feat_full["term"].isin(selected_terms))
    & (df_feat_full["loan_status"].isin(selected_status))
]


# SIDEBAR — KEY METRICS (KPIs)
# Display a quick summary of the filtered loans.

st.sidebar.markdown("---")
st.sidebar.header("Key Metrics")

# Check whether the filtered dataset contains any rows.
has_data = not filtered_df.empty

# Create two columns for the KPI cards.
left_col, right_col = st.sidebar.columns(2)

# LEFT COLUMN

with left_col:

    # Percentage of loans that defaulted.
    st.metric(
        label="Default Rate",
        value=(
            f"{filtered_df['target'].mean() * 100:.1f}%"
            if has_data else "N/A"
        )
    )

    # Average loan amount.
    st.metric(
        label="Average Loan Amount",
        value=(
            f"${filtered_df['loan_amnt'].mean():,.0f}"
            if has_data else "N/A"
        )
    )


# RIGHT COLUMN

with right_col:

    # Average interest rate.
    st.metric(
        label="Average Interest Rate",
        value=(
            f"{filtered_df['int_rate'].mean():.2f}%"
            if has_data else "N/A"
        )
    )

    # Average Debt-to-Income ratio.
    st.metric(
        label="Average DTI",
        value=(
            f"{filtered_df['dti'].mean():.1f}"
            if has_data else "N/A"
        )
    )


# CHECK FILTER RESULTS
# Stop the dashboard if no loans match the selected filters.

if not has_data:
    st.warning(
        "No loans match the selected filters. "
        "Please change the filter options and try again."
    )
    st.stop()

# CREATE DASHBOARD TABS
tab_report, tab_overview, tab_trends, tab_risk, tab_predictor, tab_models = st.tabs([
    "Report",
    "Overview",
    "Trends",
    "Risk Factors",
    "Loan Predictor",
    "Model Performance",
])


# TAB — REPORT
with tab_report:

    st.header("Report")

    # DATASET SUMMARY
    # Show general information about the dataset.

    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    # Basic dataset information
    with col1:
        st.write(f"**Total Loans:** {len(df_feat_full):,}")
        st.write(f"**Filtered Loans:** {len(filtered_df):,}")

    # Dataset categories
    with col2:
        st.write(f"**Loan Grades:** {df_feat_full['grade'].nunique()}")
        st.write(f"**Loan Purposes:** {df_feat_full['purpose'].nunique()}")

    # Date range and home ownership
    with col3:

        if (
            "issue_d" in df_feat_full.columns
            and df_feat_full["issue_d"].notna().any()
        ):

            start_date = (
                df_feat_full["issue_d"]
                .min()
                .strftime("%Y-%m-%d")
            )

            end_date = (
                df_feat_full["issue_d"]
                .max()
                .strftime("%Y-%m-%d")
            )

            st.write(f"**Loan Period:** {start_date} to {end_date}")

        st.write(
            f"**Home Ownership Types:** "
            f"{df_feat_full['home_ownership'].nunique()}"
        )

    # LOAN STATUS SUMMARY
    # Display the number of loans in each loan status.

    st.markdown("---")
    st.subheader("Loan Status Breakdown")

    status_report = filtered_df["loan_status"].value_counts()

    status_cols = st.columns(len(status_report))

    for i, (status, count) in enumerate(status_report.items()):
        status_cols[i].metric(
            label=status,
            value=f"{count:,}"
        )

    # LOANS BY GRADE
    # Display the number of loans and the default rate for each grade.

    st.markdown("---")
    st.subheader("Loans by Grade")

    grade_report = (
        filtered_df["grade"]
        .value_counts()
        .reindex(GRADE_ORDER)
        .dropna()
    )

    grade_table = pd.DataFrame({

        "Grade": grade_report.index,

        "Number of Loans": grade_report.values.astype(int),

        "Default Rate (%)": [

            round(
                filtered_df.loc[
                    filtered_df["grade"] == grade,
                    "target"
                ].mean() * 100,
                2,
            )

            for grade in grade_report.index
        ],
    })

    st.dataframe(
        grade_table,
        use_container_width=True
    )

    # TOP LOAN PURPOSES
    # Show the most common loan purposes.

    st.markdown("---")
    st.subheader("Top 10 Loan Purposes")

    purpose_report = (
        filtered_df["purpose"]
        .value_counts()
        .head(10)
    )

    purpose_table = pd.DataFrame({

        "Purpose": purpose_report.index,

        "Number of Loans": purpose_report.values,

        "Percentage (%)": (
            purpose_report.values / len(filtered_df) * 100
        ).round(2),
    })

    st.dataframe(
        purpose_table,
        use_container_width=True
    )

    # SAMPLE DATA
    # Display a sample of the filtered dataset.

    st.markdown("---")
    st.subheader("Sample Loans")

    display_columns = [

        "issue_d",
        "grade",
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "fico_range_low",
        "purpose",
        "loan_status",

    ]

    display_columns = [
        column
        for column in display_columns
        if column in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[display_columns].head(50),
        use_container_width=True,
    )


# TAB — OVERVIEW
# Explore the loan portfolio using summary charts and key insights.

with tab_overview:

    st.header("Overview")

    # Create two columns for the first row of charts.
    left_col, right_col = st.columns(2)

    # LOANS BY GRADE
    # Display the number of loans in each grade.

    with left_col:

        grade_counts = (
            filtered_df["grade"]
            .value_counts()
            .reindex(GRADE_ORDER)
            .dropna()
        )

        fig_grade = px.bar(
            x=grade_counts.index,
            y=grade_counts.values,
            title="Loans by Grade",
            labels={
                "x": "Loan Grade",
                "y": "Number of Loans",
            },
            color=grade_counts.values,
            color_continuous_scale="Viridis",
        )

        st.plotly_chart(
            fig_grade,
            use_container_width=True,
        )

        # Calculate the percentage of Grade A, B, and C loans.
        top_grades_pct = (
            filtered_df["grade"]
            .isin(["A", "B", "C"])
            .mean()
            * 100
        )

        st.info(
            f"**Grades A, B, and C account for {top_grades_pct:.1f}% "
            "of the filtered loans.**"
        )

    # LOAN STATUS DISTRIBUTION
    # Display the distribution of loan outcomes.

    with right_col:

        status_counts = filtered_df["loan_status"].value_counts()

        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Loan Status Distribution",
            color=status_counts.index,
            color_discrete_map={
                "Fully Paid": "#2ECC71",
                "Charged Off": "#E74C3C",
            },
        )

        st.plotly_chart(
            fig_status,
            use_container_width=True,
        )

        default_rate = filtered_df["target"].mean() * 100

        st.info(
            f"**Overall Default Rate: {default_rate:.1f}%**"
        )

    # Create two columns for the second row of charts.
    left_col, right_col = st.columns(2)

    # TOP LOAN PURPOSES
    # Display the 10 most common loan purposes.

    purpose_counts = (
        filtered_df["purpose"]
        .value_counts()
        .head(10)
    )

    fig_purpose = px.bar(
        x=purpose_counts.values,
        y=purpose_counts.index,
        title="Top 10 Loan Purposes",
        labels={
            "x": "Number of Loans",
            "y": "Loan Purpose",
        },
        orientation="h",
        color=purpose_counts.values,
        color_continuous_scale="Blues",
    )

    st.plotly_chart(
        fig_purpose,
        use_container_width=True,
    )

    top3_purposes = purpose_counts.head(3)

    top3_percentage = (
        top3_purposes.sum()
        / len(filtered_df)
        * 100
    )

    st.info(
        f"**Top 3 purposes ({', '.join(top3_purposes.index)}) "
        f"account for {top3_percentage:.1f}% of the filtered loans.**"
    )

    # HOME OWNERSHIP DISTRIBUTION
    # Display the distribution of home ownership.

    home_counts = (filtered_df["home_ownership"].
                    where(filtered_df["home_ownership"].
                            isin(["RENT", "MORTGAGE", "OWN", "OTHER"]))
                    .value_counts()
    )

    fig_home = px.pie(
        values=home_counts.values,
        names=home_counts.index,
        title="Home Ownership Distribution",
    )

    st.plotly_chart(
        fig_home,
        use_container_width=True,
    )

    st.metric(
    label="Most Common Home Ownership",
    value=home_counts.idxmax(),
    delta=f"{home_counts.max():,} loans"
    )

# TAB — TRENDS OVER TIME
# Analyze how Lending Club loan activity changes over time.

with tab_trends:

    st.header("Trends Over Time")

    # Continue only if the issue date is available.
    if (
        "issue_d" in filtered_df.columns
        and filtered_df["issue_d"].notna().any()
    ):

        # Create a copy of the filtered data and remove
        # rows with missing issue dates.
        trend_df = (
            filtered_df
            .dropna(subset=["issue_d"])
            .copy()
        )

        # Group loans by year and month.
        trend_df["year_month"] = (
            trend_df["issue_d"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        # Create two columns for the first row of charts.
        left_col, right_col = st.columns(2)

        # ====================================================================
        # MONTHLY LOAN VOLUME
        # ====================================================================
        # Show how many loans were issued each month.

        with left_col:

            monthly_loans = (
                trend_df
                .groupby("year_month")
                .size()
            )

            fig_volume = go.Figure(
                data=[
                    go.Scatter(
                        x=monthly_loans.index,
                        y=monthly_loans.values,
                        mode="lines+markers",
                        fill="tozeroy",
                        line=dict(color="#1F77B4"),
                    )
                ]
            )

            fig_volume.update_layout(
                title="Monthly Loan Volume",
                xaxis_title="Month",
                yaxis_title="Number of Loans",
                hovermode="x unified",
            )

            st.plotly_chart(
                fig_volume,
                use_container_width=True,
            )

        # ====================================================================
        # MONTHLY DEFAULT RATE
        # ====================================================================
        # Show how the default rate changes over time.

        with right_col:

            monthly_default_rate = (
                trend_df
                .groupby("year_month")["target"]
                .mean()
                * 100
            )

            fig_default = go.Figure(
                data=[
                    go.Scatter(
                        x=monthly_default_rate.index,
                        y=monthly_default_rate.values,
                        mode="lines+markers",
                        fill="tozeroy",
                        line=dict(color="#E74C3C"),
                    )
                ]
            )

            fig_default.update_layout(
                title="Monthly Default Rate",
                xaxis_title="Month",
                yaxis_title="Default Rate (%)",
                hovermode="x unified",
            )

            st.plotly_chart(
                fig_default,
                use_container_width=True,
            )

        # ====================================================================
        # MONTHLY AVERAGE LOAN AMOUNT
        # ====================================================================
        # Show how the average loan amount changes over time.

        monthly_avg_loan = (
            trend_df
            .groupby("year_month")["loan_amnt"]
            .mean()
        )

        fig_amount = go.Figure(
            data=[
                go.Scatter(
                    x=monthly_avg_loan.index,
                    y=monthly_avg_loan.values,
                    mode="lines+markers",
                    fill="tozeroy",
                    line=dict(color="#2ECC71"),
                )
            ]
        )

        fig_amount.update_layout(
            title="Monthly Average Loan Amount",
            xaxis_title="Month",
            yaxis_title="Average Loan Amount ($)",
            hovermode="x unified",
        )

        st.plotly_chart(
            fig_amount,
            use_container_width=True,
        )

    else:
        st.info(
            "No valid issue dates were found in the dataset, "
            "so time-based trends cannot be displayed."
        )

# ============================================================================
# TAB — RISK FACTORS
# ============================================================================
# Analyze the relationship between borrower characteristics
# and loan default risk.

with tab_risk:

    st.header("Risk Factors")

    # Create a copy to avoid modifying the filtered dataset.
    risk_df = filtered_df.copy()

    # ========================================================================
    # DEFAULT RATE BY FICO SCORE
    # ========================================================================
    # Compare default rates across different FICO score ranges.

    risk_df["fico_range"] = pd.cut(
        risk_df["fico_range_low"],
        bins=[600, 660, 700, 740, 780, 850],
        labels=["620–660", "660–700", "700–740", "740–780", "780+"],
    )

    fico_default = (
        risk_df
        .groupby("fico_range", observed=True)["target"]
        .mean()
        * 100
    )

    fig_fico = px.bar(
        x=fico_default.index.astype(str),
        y=fico_default.values,
        title="Default Rate by FICO Score",
        labels={
            "x": "FICO Score Range",
            "y": "Default Rate (%)",
        },
        color=fico_default.values,
        color_continuous_scale="RdYlGn_r",
    )

    st.plotly_chart(
        fig_fico,
        use_container_width=True,
    )

    st.info(
            f"Average FICO Score: {filtered_df['fico_range_low'].mean():.0f}"
    )
    # Create two columns for the remaining charts.
    left_col, right_col = st.columns(2)

    # ========================================================================
    # DEFAULT RATE BY DTI
    # ========================================================================
    # Compare default rates across Debt-to-Income (DTI) ranges.

    with left_col:

        risk_df["dti_range"] = pd.cut(
            risk_df["dti"],
            bins=[-0.1, 10, 20, 30, 45],
            labels=["0–10", "10–20", "20–30", "30–45"],
        )

        dti_default = (
            risk_df
            .groupby("dti_range", observed=True)["target"]
            .mean()
            * 100
        )

        fig_dti = px.bar(
            x=dti_default.index.astype(str),
            y=dti_default.values,
            title="Default Rate by DTI",
            labels={
                "x": "DTI Range",
                "y": "Default Rate (%)",
            },
            color=dti_default.values,
            color_continuous_scale="Oranges",
        )

        st.plotly_chart(
            fig_dti,
            use_container_width=True,
        )

        # Percentage of loans with high DTI.
        high_dti_pct = (
            (risk_df["dti"] > 25)
            .mean()
            * 100
        )

        st.info(
            f"**High DTI (>25): {high_dti_pct:.0f}%** of filtered loans."
        )

    # ========================================================================
    # DEFAULT RATE BY CREDIT UTILIZATION
    # ========================================================================
    # Compare default rates across revolving credit utilization ranges.

    with right_col:

        risk_df["utilization_range"] = pd.cut(
            risk_df["revol_util"],
            bins=[-0.1, 30, 50, 70, 100, 200],
            labels=["0–30", "30–50", "50–70", "70–100", "100+"],
        )

        utilization_default = (
            risk_df
            .groupby("utilization_range", observed=True)["target"]
            .mean()
            * 100
        )

        fig_utilization = px.bar(
            x=utilization_default.index.astype(str),
            y=utilization_default.values,
            title="Default Rate by Credit Utilization",
            labels={
                "x": "Credit Utilization (%)",
                "y": "Default Rate (%)",
            },
            color=utilization_default.values,
            color_continuous_scale="Reds",
        )

        st.plotly_chart(
            fig_utilization,
            use_container_width=True,
        )

        # Percentage of loans with high credit utilization.
        high_utilization_pct = (
            (risk_df["revol_util"] > 70)
            .mean()
            * 100
        )

        st.info(
            f"**High Credit Utilization (>70%): {high_utilization_pct:.0f}%** of filtered loans."
        )

    # ============================================================================
    # FEATURE CORRELATION HEATMAP
    # ============================================================================

    st.markdown("---")
    st.subheader("🔗 Feature Correlation Heatmap")

    # Select numerical columns
    numeric_columns = [
        column
        for column in NUMERIC_BASE_COLS + ["target"]
        if column in risk_df.columns
    ]

    # Calculate correlation matrix
    correlation_matrix = risk_df[numeric_columns].corr()

    # Create heatmap
    fig_corr = px.imshow(
        correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        text_auto=".2f",              # Show correlation values
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    # HIGHLIGHT STRONG CORRELATIONS

    highlight_threshold = 0.40
    features = correlation_matrix.columns.tolist()

    for i in range(len(features)):
        for j in range(i + 1, len(features)):

            correlation_value = correlation_matrix.iloc[i, j]

            if abs(correlation_value) >= highlight_threshold:

                fig_corr.add_shape(
                    type="rect",
                    x0=j - 0.48,
                    x1=j + 0.48,
                    y0=i - 0.48,
                    y1=i + 0.48,
                    line=dict(
                        color="black",
                        width=3
                    )
                )

                # Highlight the symmetric cell too
                fig_corr.add_shape(
                    type="rect",
                    x0=i - 0.48,
                    x1=i + 0.48,
                    y0=j - 0.48,
                    y1=j + 0.48,
                    line=dict(
                        color="black",
                        width=3
                    )
                )

    fig_corr.update_layout(
        height=650,
        xaxis_title="Features",
        yaxis_title="Features"
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

    st.info(
        "**Highlighted cells** show strong correlations "
        f"(absolute correlation ≥ {highlight_threshold:.2f})."
    )


# ============================================================================
# TRAIN MACHINE LEARNING MODELS
# ============================================================================
# Train all models once and reuse them
# in the Predictor and Model Performance tabs.

bundle = train_models(df_feat_full)

results = bundle["results"]
best_name = bundle["best_name"]

# ============================================================================
# TAB 5 — LOAN DEFAULT PREDICTOR
# ============================================================================
# Enter borrower information and use the trained
# machine learning model to predict loan default risk.

with tab_predictor:

    st.header("Loan Default Predictor")

    st.write(
        "The selected model will estimate the probability that the loan will default."
    )

    # ========================================================================
    # SELECT MACHINE LEARNING MODEL
    # ========================================================================
    # Choose which trained model to use.

    model_choice = st.selectbox(
        "Prediction Model",
        list(results.keys()),
        index=list(results.keys()).index(best_name)
    )

    # Display model evaluation metrics.
    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "F1 Score",
            f"{results[model_choice]['F1']:.3f}"
        )

    with metric_col2:
        st.metric(
            "ROC-AUC",
            f"{results[model_choice]['ROC-AUC']:.3f}"
        )

    # ========================================================================
    # INPUT FORM
    # ========================================================================
    # Collect borrower and loan information.

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        # --------------------------------------------------------------------
        # LOAN INFORMATION
        # --------------------------------------------------------------------

        with col1:

            loan_amnt = st.number_input(
                "Loan Amount ($)",
                1000,
                40000,
                15000,
                step=500
            )

            term = st.selectbox(
                "Loan Term (Months)",
                [36, 60]
            )

            int_rate = st.slider(
                "Interest Rate (%)",
                5.0,
                31.0,
                13.5,
                step=0.1
            )

            installment = st.number_input(
                "Monthly Installment ($)",
                30.0,
                2000.0,
                450.0,
                step=10.0
            )


        # --------------------------------------------------------------------
        # BORROWER INFORMATION
        # --------------------------------------------------------------------

        with col2:

            annual_inc = st.number_input(
                "Annual Income ($)",
                10000,
                500000,
                60000,
                step=1000
            )

            emp_length_label = st.selectbox(
                "Employment Length",
                list(EMP_LENGTH_OPTS.keys()),
                index=10
            )

            dti = st.slider(
                "Debt-to-Income Ratio",
                0.0,
                45.0,
                18.0,
                step=0.5
            )

            grade = st.selectbox(
                "Loan Grade",
                GRADE_ORDER,
                index=2
            )

        # --------------------------------------------------------------------
        # CREDIT HISTORY
        # --------------------------------------------------------------------

        with col3:

            fico_range_low = st.slider(
                "FICO Score",
                620,
                845,
                700,
                step=5
            )

            open_acc = st.number_input(
                "Open Credit Accounts",
                0,
                40,
                10
            )

            pub_rec = st.number_input(
                "Public Records",
                0,
                5,
                0
            )

            revol_util = st.slider(
                "Credit Utilization (%)",
                0.0,
                150.0,
                45.0,
                step=1.0
            )

        # Predict button.
        submitted = st.form_submit_button(
            "Predict",
            use_container_width=True
        )

    # ========================================================================
    # MAKE PREDICTION
    # ========================================================================

    if submitted:

        # Store all user inputs in a dictionary.
        input_data = {
            "loan_amnt": loan_amnt,
            "term": term,
            "int_rate": int_rate,
            "installment": installment,
            "grade": grade,
            "emp_length": EMP_LENGTH_OPTS[emp_length_label],
            "annual_inc": annual_inc,
            "dti": dti,
            "fico_range_low": fico_range_low,
            "open_acc": open_acc,
            "pub_rec": pub_rec,
            "revol_util": revol_util,
        }

        prediction, probability = predict_single(
            input_data,
            bundle,
            model_choice
        )

        st.markdown("---")

        result_col, gauge_col = st.columns([1, 1.3])

        # --------------------------------------------------------------------
        # PREDICTION RESULT
        # --------------------------------------------------------------------

        with result_col:

            if prediction == 1:
                st.error("### ⚠️ High Risk of Default")

            else:
                st.success("### ✅ Likely to be Fully Paid")

            st.write(
                f"**Default Probability:** {probability:.1%}"
            )

            st.caption(
                f"Model Used: **{model_choice}** "
                "(Decision Threshold = 50%)"
            )

            # Explain the main risk factors.
            risk_factors = []

            if int_rate > 18:
                risk_factors.append(
                    "High interest rate."
                )

            if fico_range_low < 660:
                risk_factors.append(
                    "Low FICO score."
                )

            if dti > 25:
                risk_factors.append(
                    "High debt-to-income ratio."
                )

            if revol_util > 70:
                risk_factors.append(
                    "High credit utilization."
                )

            if pub_rec > 0:
                risk_factors.append(
                    "Public derogatory records found."
                )

            if risk_factors:

                st.write("### Main Risk Factors")

                for factor in risk_factors:
                    st.write(f"- {factor}")

        # --------------------------------------------------------------------
        # RISK GAUGE
        # --------------------------------------------------------------------

        with gauge_col:

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={"text": "Default Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2C3E50"},
                        "steps": [
                            {"range": [0, 30], "color": "#2ECC71"},
                            {"range": [30, 60], "color": "#F39C12"},
                            {"range": [60, 100], "color": "#E74C3C"},
                        ],
                        "threshold": {
                            "line": {
                                "color": "black",
                                "width": 3
                            },
                            "value": 50,
                        },
                    },
                )
            )

            fig_gauge.update_layout(
                height=300,
                margin=dict(t=40, b=10)
            )

            st.plotly_chart(
                fig_gauge,
                use_container_width=True
            )


# ============================================================================
# TAB 6 — MODEL PERFORMANCE
# ============================================================================
# Compare the performance of all trained machine learning models.

with tab_models:

    st.header("Model Performance")

    # ========================================================================
    # MODEL COMPARISON TABLE
    # ========================================================================
    # Display the evaluation metrics for each model.

    model_summary = pd.DataFrame([
        {
            "Model": model_name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1"],
            "ROC-AUC": metrics["ROC-AUC"],
        }

        for model_name, metrics in results.items()
    ]).set_index("Model").round(4)

    st.dataframe(
        model_summary,
        use_container_width=True
    )

    st.success(
        f"Best Model: **{best_name}** "
        f"(Highest F1 Score)"
    )

    # Create two columns for the performance charts.
    col1, col2 = st.columns(2)

    # ========================================================================
    # ROC CURVES
    # ========================================================================
    # Compare the ROC curve for every trained model.

    with col1:

        fig_roc = go.Figure()

        for model_name, metrics in results.items():

            fpr, tpr, _ = roc_curve(
                metrics["y_test"],
                metrics["y_prob"]
            )

            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{model_name} (AUC = {metrics['ROC-AUC']:.3f})"
                )
            )

        # Add a random prediction reference line.
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(
                    dash="dash",
                    color="black"
                ),
                name="Random Guess"
            )
        )

        fig_roc.update_layout(
            title="ROC Curve Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white"
        )

        st.plotly_chart(
            fig_roc,
            use_container_width=True
        )

    # ========================================================================
    # CONFUSION MATRIX
    # ========================================================================
    # Display the confusion matrix for the selected model.

    with col2:

        selected_model = st.selectbox(
            "Select a Model",
            list(results.keys()),
            index=list(results.keys()).index(best_name),
            key="cm_select"
        )

        confusion = confusion_matrix(
            results[selected_model]["y_test"],
            results[selected_model]["y_pred"]
        )

        fig_confusion = px.imshow(
            confusion,
            text_auto=True,
            x=["Fully Paid", "Default"],
            y=["Fully Paid", "Default"],
            labels={
                "x": "Predicted",
                "y": "Actual"
            },
            color_continuous_scale="Blues",
            title=f"Confusion Matrix - {selected_model}"
        )

        st.plotly_chart(
            fig_confusion,
            use_container_width=True
        )

    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================
    # Display the most important features
    # if the selected model supports feature importance.

    best_model = results[best_name]["model"]

    if hasattr(best_model, "feature_importances_"):

        feature_importance = pd.DataFrame({
            "Feature": bundle["feature_cols"],
            "Importance": best_model.feature_importances_,
        })

        feature_importance = (
            feature_importance
            .sort_values("Importance", ascending=False)
            .head(15)
        )

        fig_importance = px.bar(
            feature_importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top 15 Most Important Features ({best_name})",
            color="Importance",
            color_continuous_scale="Viridis"
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True
        )
