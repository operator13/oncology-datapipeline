"""
Streamlit dashboard for oncology data quality metrics.

Run with: streamlit run dashboards/quality_dashboard.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Oncology Data Quality Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main dashboard application."""
    st.title("📊 Oncology Data Quality Dashboard")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Data Profiling", "Validation Results", "Anomaly Detection", "Generate Data"],
    )

    if page == "Overview":
        render_overview()
    elif page == "Data Profiling":
        render_profiling()
    elif page == "Validation Results":
        render_validation()
    elif page == "Anomaly Detection":
        render_anomalies()
    elif page == "Generate Data":
        render_generator()


def render_overview():
    """Render overview page."""
    st.header("Data Quality Overview")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Overall Quality Score",
            value="92.5%",
            delta="2.3%",
        )

    with col2:
        st.metric(
            label="Completeness",
            value="98.2%",
            delta="0.5%",
        )

    with col3:
        st.metric(
            label="Validity",
            value="95.1%",
            delta="-0.3%",
        )

    with col4:
        st.metric(
            label="Anomalies Detected",
            value="23",
            delta="-5",
        )

    st.markdown("---")

    # Quality dimensions chart
    st.subheader("Quality Dimensions")

    dimensions_data = pd.DataFrame({
        "Dimension": ["Completeness", "Accuracy", "Consistency", "Timeliness", "Validity", "Uniqueness"],
        "Score": [98.2, 94.5, 91.3, 89.7, 95.1, 99.8],
    })

    st.bar_chart(dimensions_data.set_index("Dimension"))

    # Recent validations
    st.subheader("Recent Validation Runs")

    validation_history = pd.DataFrame({
        "Timestamp": ["2024-01-15 10:00", "2024-01-14 10:00", "2024-01-13 10:00"],
        "Dataset": ["patients", "treatments", "lab_results"],
        "Status": ["✅ Passed", "✅ Passed", "⚠️ Warning"],
        "Score": [95.2, 92.1, 78.5],
    })

    st.dataframe(validation_history, use_container_width=True)


def render_profiling():
    """Render data profiling page."""
    st.header("Data Profiling")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file to profile", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        st.subheader("Column Statistics")

        # Calculate basic stats
        stats_data = []
        for col in df.columns:
            null_count = df[col].isna().sum()
            null_pct = null_count / len(df) * 100
            unique_count = df[col].nunique()

            stats_data.append({
                "Column": col,
                "Type": str(df[col].dtype),
                "Non-Null": len(df) - null_count,
                "Null %": f"{null_pct:.1f}%",
                "Unique": unique_count,
            })

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)

        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

    else:
        st.info("Upload a CSV file to see profiling results")


def render_validation():
    """Render validation results page."""
    st.header("Validation Results")

    # Dataset selector
    dataset = st.selectbox(
        "Select Dataset",
        ["patients", "treatments", "lab_results"],
    )

    st.subheader(f"Validation Results: {dataset}")

    # Mock validation results
    expectations = [
        ("expect_column_values_to_be_unique", "patient_id", "✅ Passed", 100.0),
        ("expect_column_values_to_not_be_null", "mrn", "✅ Passed", 100.0),
        ("expect_column_values_to_match_regex", "mrn", "✅ Passed", 99.8),
        ("expect_column_values_to_be_in_set", "gender", "✅ Passed", 100.0),
        ("expect_column_values_to_match_regex", "diagnosis_code", "⚠️ Warning", 95.2),
        ("expect_column_pair_values_A_greater_than_B", "diagnosis_date > birth_date", "✅ Passed", 100.0),
    ]

    results_df = pd.DataFrame(
        expectations,
        columns=["Expectation", "Column/Condition", "Status", "Success %"],
    )

    st.dataframe(results_df, use_container_width=True)

    # Summary metrics
    passed = sum(1 for e in expectations if "Passed" in e[2])
    total = len(expectations)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Expectations Passed", f"{passed}/{total}")
    with col2:
        st.metric("Overall Success Rate", f"{passed/total*100:.1f}%")


def render_anomalies():
    """Render anomaly detection page."""
    st.header("Anomaly Detection")

    st.subheader("Configuration")

    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox(
            "Detection Method",
            ["IQR", "Z-Score", "MAD", "Isolation Forest"],
        )
    with col2:
        threshold = st.slider("Threshold", 1.0, 5.0, 1.5, 0.1)

    st.subheader("Detected Anomalies")

    # Mock anomaly data
    anomalies_df = pd.DataFrame({
        "Record ID": ["P-001234", "P-002345", "P-003456", "P-004567"],
        "Column": ["lab_value", "age", "lab_value", "dosage"],
        "Value": [1250.0, -5, 0.001, 50000],
        "Expected Range": ["0-100", "18-120", "0-100", "0-1000"],
        "Anomaly Score": [0.95, 0.88, 0.92, 0.99],
    })

    st.dataframe(anomalies_df, use_container_width=True)

    st.metric("Total Anomalies Detected", len(anomalies_df))


def render_generator():
    """Render data generator page."""
    st.header("Synthetic Data Generator")

    st.markdown("""
    Generate synthetic oncology data for testing and development.
    The generator creates realistic patient demographics, treatment records,
    and laboratory results.
    """)

    col1, col2 = st.columns(2)

    with col1:
        num_patients = st.number_input(
            "Number of Patients",
            min_value=10,
            max_value=10000,
            value=100,
            step=10,
        )

    with col2:
        seed = st.number_input(
            "Random Seed (for reproducibility)",
            min_value=0,
            max_value=9999,
            value=42,
        )

    cancer_types = st.multiselect(
        "Cancer Types",
        ["breast", "lung", "colorectal", "prostate", "melanoma", "lymphoma"],
        default=["breast", "lung"],
    )

    if st.button("Generate Data", type="primary"):
        with st.spinner("Generating synthetic data..."):
            try:
                from src.synthetic_data import OncologyDataFactory, CancerType

                # Map string to enum
                type_map = {
                    "breast": CancerType.BREAST,
                    "lung": CancerType.LUNG,
                    "colorectal": CancerType.COLORECTAL,
                    "prostate": CancerType.PROSTATE,
                    "melanoma": CancerType.MELANOMA,
                    "lymphoma": CancerType.LYMPHOMA,
                }

                selected_types = [type_map[t] for t in cancer_types]

                factory = OncologyDataFactory(
                    num_patients=num_patients,
                    seed=seed,
                    cancer_types=selected_types,
                )
                dataset = factory.generate()

                st.success(f"Generated {dataset.patient_count} patients!")

                # Show summary
                st.json(dataset.summary())

                # Download buttons
                dfs = dataset.to_dataframes()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        "Download Patients CSV",
                        dfs["patients"].to_csv(index=False),
                        "patients.csv",
                        "text/csv",
                    )
                with col2:
                    st.download_button(
                        "Download Treatments CSV",
                        dfs["treatments"].to_csv(index=False),
                        "treatments.csv",
                        "text/csv",
                    )
                with col3:
                    st.download_button(
                        "Download Lab Results CSV",
                        dfs["lab_results"].to_csv(index=False),
                        "lab_results.csv",
                        "text/csv",
                    )

            except Exception as e:
                st.error(f"Error generating data: {str(e)}")


if __name__ == "__main__":
    main()
