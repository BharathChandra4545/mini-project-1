
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Student Learning Analytics",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = Path("models/risk_model.pkl")

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #1c2333;
        border: 1px solid #343e52;
        padding: 18px;
        border-radius: 12px;
    }
    .subtitle {
        color: #aab5c8;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🎓 Student Performance & Learning-Risk Analytics")
st.markdown(
    '<p class="subtitle">Academic insights • Early-warning indicators • '
    'Machine-learning predictions</p>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- LOAD MODEL ----------------
if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Unable to load model: {exc}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📚 Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Overview Dashboard",
        "Performance Analysis",
        "Predict Learning Risk",
        "Student Report",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Student analytics project")

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_default_data():
    path = Path("data/cleaned_students.csv")
    if not path.exists():
        path = Path("data/student-mat.csv")

    if not path.exists():
        return None

    return pd.read_csv(path, sep=None, engine="python")


def read_uploaded_data(uploaded_file):
    return pd.read_csv(uploaded_file, sep=None, engine="python")


uploaded_file = st.sidebar.file_uploader(
    "Upload student CSV (optional)",
    type=["csv"]
)

try:
    if uploaded_file is not None:
        df = read_uploaded_data(uploaded_file)
    else:
        df = load_default_data()
except Exception as exc:
    st.error(f"Unable to read the dataset: {exc}")
    st.stop()

if df is None:
    st.warning(
        "Dataset not found. Put student-mat.csv inside the data folder "
        "or upload a CSV using the sidebar."
    )
    st.stop()

if df.empty:
    st.error("The dataset contains no records.")
    st.stop()

# Normalize headers and validate required fields
df.columns = df.columns.str.strip()

prediction_features = [
    "G1", "G2", "studytime", "absences", "failures", "age"
]

if "G3" in df.columns:
    df["G3"] = pd.to_numeric(df["G3"], errors="coerce")

for col in prediction_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------- COMMON HELPERS ----------------
def format_grade(value):
    return "N/A" if pd.isna(value) else f"{value:.2f}/20"


def get_predictions(data):
    missing = [col for col in prediction_features if col not in data.columns]

    if missing:
        raise ValueError("Missing required features: " + ", ".join(missing))

    inputs = data[prediction_features].copy()
    predictions = model.predict(inputs)
    probabilities = model.predict_proba(inputs)[:, 1]

    result = data.copy()
    result["Predicted_Risk"] = predictions.astype(int)
    result["Risk_Probability"] = probabilities
    result["Risk_Status"] = result["Predicted_Risk"].map({
        1: "Potential support needed",
        0: "Not flagged by model"
    })

    return result


# ---------------- OVERVIEW DASHBOARD ----------------
if page == "Overview Dashboard":
    st.subheader("📊 Academic Overview")

    total_students = len(df)
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", f"{total_students:,}")

    if "G3" in df.columns:
        valid_grades = df["G3"].dropna()
        avg_grade = valid_grades.mean() if not valid_grades.empty else np.nan
        at_risk = int((valid_grades < 10).sum())
        valid_count = len(valid_grades)
        risk_pct = at_risk / valid_count * 100 if valid_count else 0

        c2.metric("Average Final Grade", format_grade(avg_grade))
        c3.metric("Grade Below 10", at_risk)
        c4.metric("Below-10 Percentage", f"{risk_pct:.1f}%")

        st.caption(
            "The grade-below-10 figures describe observed final grades. "
            "They are not model predictions. This is an illustrative "
            "project threshold, not an official academic standard."
        )
    else:
        c2.metric("Average Final Grade", "N/A")
        c3.metric("Grade Below 10", "N/A")
        c4.metric("Below-10 Percentage", "N/A")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Final Grade Distribution")

        if "G3" in df.columns:
            grades = df["G3"].dropna().astype(int)
            grade_counts = (
                grades.value_counts()
                .reindex(range(21), fill_value=0)
                .sort_index()
            )
            grade_counts.index.name = "Final Grade"
            grade_counts.name = "Students"
            st.bar_chart(grade_counts)
        else:
            st.info("The uploaded dataset does not contain G3.")

    with right:
        st.subheader("Study Time vs Average Grade")

        if {"studytime", "G3"}.issubset(df.columns):
            study_data = df.dropna(subset=["studytime", "G3"])
            study_summary = (
                study_data.groupby("studytime")["G3"]
                .mean()
                .rename("Average Grade")
            )
            st.bar_chart(study_summary)
        else:
            st.info("Study-time or final-grade data is unavailable.")

    st.subheader("Absences vs Final Grade")

    if {"absences", "G3"}.issubset(df.columns):
        scatter_data = df[["absences", "G3"]].dropna()
        st.scatter_chart(
            scatter_data,
            x="absences",
            y="G3"
        )
    else:
        st.info("Absence or final-grade data is unavailable.")

    st.subheader("Recent Student Records")
    st.dataframe(df.head(10), use_container_width=True)

# ---------------- PERFORMANCE ANALYSIS ----------------
elif page == "Performance Analysis":
    st.subheader("📈 Performance Analysis")

    if "G3" not in df.columns:
        st.warning("This section requires the G3 final-grade column.")
        st.stop()

    valid_grades = df["G3"].dropna()

    a, b, c = st.columns(3)
    a.metric("Highest Grade", format_grade(valid_grades.max()) if not valid_grades.empty else "N/A")
    b.metric("Lowest Grade", format_grade(valid_grades.min()) if not valid_grades.empty else "N/A")
    c.metric("Median Grade", format_grade(valid_grades.median()) if not valid_grades.empty else "N/A")

    st.subheader("Grade Statistics")
    st.dataframe(
        valid_grades.describe().rename("Value").to_frame(),
        use_container_width=True
    )

    if {"G1", "G2", "G3"}.issubset(df.columns):
        st.subheader("Grades Across Periods")
        grade_means = df[["G1", "G2", "G3"]].mean()
        st.line_chart(grade_means)

        st.subheader("Second-Period Grade vs Final Grade")
        st.scatter_chart(df[["G2", "G3"]].dropna(), x="G2", y="G3")

    st.subheader("Numerical Correlation Heatmap")
    numeric = df.select_dtypes(include="number")

    if len(numeric.columns) >= 2:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(
            numeric.corr(),
            cmap="coolwarm",
            center=0,
            ax=ax
        )
        ax.set_title("Correlation Between Numerical Variables")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Not enough numerical columns for correlation analysis.")

# ---------------- PREDICT LEARNING RISK ----------------
elif page == "Predict Learning Risk":
    st.subheader("🧠 Individual Student Risk Prediction")

    st.write(
        "Estimate whether a student's final grade may fall below 10/20 "
        "based on earlier grades and selected student information."
    )

    st.info(
        "This model is a prototype for educational analysis. "
        "A prediction is not a diagnosis or a definitive judgment."
    )

    with st.form("risk_form"):
        col1, col2 = st.columns(2)

        with col1:
            g1 = st.number_input("First-period grade (G1)", 0, 20, 10)
            g2 = st.number_input("Second-period grade (G2)", 0, 20, 10)
            studytime = st.selectbox(
                "Weekly study-time category",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1 — Less than 2 hours",
                    2: "2 — 2 to 5 hours",
                    3: "3 — 5 to 10 hours",
                    4: "4 — More than 10 hours"
                }[x]
            )

        with col2:
            absences = st.number_input("Absences", min_value=0, value=0)
            failures = st.number_input("Past class failures", min_value=0, value=0)
            age = st.number_input("Age", min_value=15, max_value=30, value=17)

        submitted = st.form_submit_button(
            "Predict Learning Risk",
            type="primary"
        )

    if submitted:
        student = pd.DataFrame([{
            "G1": g1,
            "G2": g2,
            "studytime": studytime,
            "absences": absences,
            "failures": failures,
            "age": age
        }])

        try:
            prediction = int(model.predict(student)[0])
            probability = float(model.predict_proba(student)[0, 1])

            st.divider()
            r1, r2 = st.columns(2)
            r1.metric("Estimated Risk Probability", f"{probability:.1%}")
            r2.metric(
                "Model Classification",
                "Flagged for review" if prediction == 1 else "Not flagged"
            )

            st.progress(probability)

            if prediction == 1:
                st.warning(
                    "Consider reviewing this student's progress and "
                    "whether additional academic support could help."
                )
            else:
                st.success(
                    "The model did not flag this input. Continue normal "
                    "academic monitoring."
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# ---------------- STUDENT REPORT ----------------
elif page == "Student Report":
    st.subheader("📄 Student Report")

    if st.button("Generate Risk Report", type="primary"):
        try:
            report = get_predictions(df)

            st.success("Report generated.")

            flagged_count = int((report["Predicted_Risk"] == 1).sum())
            r1, r2 = st.columns(2)
            r1.metric("Students Analyzed", len(report))
            r2.metric("Flagged for Review", flagged_count)

            st.dataframe(report, use_container_width=True)

            csv = report.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Student Risk Report (CSV)",
                data=csv,
                file_name="student_learning_risk_report.csv",
                mime="text/csv"
            )

            st.caption(
                "Predictions are estimates. Review individual cases before "
                "making educational decisions."
            )

        except Exception as exc:
            st.error(f"Unable to generate report: {exc}")

    else:
        st.write("Click the button to generate predictions for the loaded dataset.")

# ---------------- ABOUT PROJECT ----------------
elif page == "About Project":
    st.subheader("ℹ️ About This Project")

    st.markdown("""
    **Objective**

    Analyze academic performance and explore early indicators that may
    help identify students who could benefit from additional support.

    **Technologies used**

    - Python
    - NumPy and Pandas
    - Matplotlib and Seaborn
    - Scikit-learn
    - Joblib
    - Streamlit

    **Machine-learning model**

    Logistic Regression trained using earlier-period grades and selected
    student features.

    **Risk definition**

    A final grade below 10 out of 20 is the target used for this prototype.

    **Limitations**

    The model is trained on a small public dataset. Results may not
    generalize to other schools or student populations. Model estimates
    should not be used as the sole basis for educational decisions.
    """)
