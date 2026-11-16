import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import hashlib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Student Performance Insights",
    page_icon="🎓",
    layout="wide"
)

# =====================================================
# CUSTOM LIGHT THEME
# =====================================================

st.markdown("""
<style>
    .stApp {
        background-color: #F7F8F5;
        color: #26352D;
    }

    [data-testid="stSidebar"] {
        background-color: #EEF2EC;
    }

    .main-title {
        font-size: 34px;
        font-weight: 750;
        color: #294C3A;
    }

    .sub-title {
        color: #718078;
        font-size: 15px;
    }

    .student-card {
        background: white;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #E5EBE5;
        margin-bottom: 12px;
    }

    .student-name {
        font-size: 19px;
        font-weight: bold;
        color: #294C3A;
    }

    .student-info {
        color: #718078;
        font-size: 14px;
    }

    .success-text {
        color: #218653;
        font-weight: bold;
    }

    .danger-text {
        color: #D14D4D;
        font-weight: bold;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 17px;
        border-radius: 14px;
        border: 1px solid #E5EBE5;
    }

    .section-heading {
        color: #294C3A;
        font-size: 23px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# CONSTANTS
# =====================================================

PASS_MARK = 40

FEATURES = [
    "Attendance",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Marks",
    "Study_Hours"
]

REQUIRED_TRAINING_COLUMNS = FEATURES + ["Final_Marks"]

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "student_performance.csv"
)


# =====================================================
# LOAD AND CLEAN TRAINING DATA
# =====================================================

@st.cache_data
def load_training_data():

    if not os.path.exists(DATASET_PATH):
        return None

    data = pd.read_csv(DATASET_PATH)

    missing = [
        column for column in REQUIRED_TRAINING_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Training dataset is missing these columns: "
            + ", ".join(missing)
        )

    data = data[REQUIRED_TRAINING_COLUMNS].copy()

    for column in REQUIRED_TRAINING_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(subset=["Final_Marks"])

    if len(data) < 2:
        raise ValueError(
            "At least two valid student records are required "
            "in the training dataset."
        )

    # Fill missing feature values using median
    for column in FEATURES:
        median_value = data[column].median()

        if pd.isna(median_value):
            median_value = 0

        data[column] = data[column].fillna(median_value)

    # Keep values within reasonable ranges
    for column in [
        "Attendance",
        "Assignment_Score",
        "Internal_Marks",
        "Previous_Marks",
        "Final_Marks"
    ]:
        data[column] = data[column].clip(0, 100)

    data["Study_Hours"] = data["Study_Hours"].clip(0, 24)

    return data


# =====================================================
# TRAIN MACHINE LEARNING MODELS
# =====================================================

@st.cache_resource
def train_models(data):

    X = data[FEATURES]

    y_marks = data["Final_Marks"]

    y_result = (
        data["Final_Marks"] >= PASS_MARK
    ).astype(int)

    # Predict final marks
    regressor = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    regressor.fit(X, y_marks)

    # Predict pass/fail
    classifier = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    classifier.fit(X, y_result)

    return regressor, classifier


# =====================================================
# CLEAN UPLOADED CSV
# =====================================================

def prepare_uploaded_data(uploaded_data, training_data):

    uploaded_data = uploaded_data.copy()

    missing_columns = [
        column for column in FEATURES
        if column not in uploaded_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Your CSV is missing these columns: "
            + ", ".join(missing_columns)
        )

    # Create Student_ID if it does not exist
    if "Student_ID" not in uploaded_data.columns:
        uploaded_data.insert(
            0,
            "Student_ID",
            [f"Student {i+1}" for i in range(len(uploaded_data))]
        )

    # Convert feature columns into numbers
    for column in FEATURES:
        uploaded_data[column] = pd.to_numeric(
            uploaded_data[column],
            errors="coerce"
        )

        # Fill missing values using training dataset median
        median_value = training_data[column].median()

        uploaded_data[column] = uploaded_data[column].fillna(
            median_value
        )

    # Limit values
    for column in [
        "Attendance",
        "Assignment_Score",
        "Internal_Marks",
        "Previous_Marks"
    ]:
        uploaded_data[column] = uploaded_data[column].clip(0, 100)

    uploaded_data["Study_Hours"] = uploaded_data[
        "Study_Hours"
    ].clip(0, 24)

    return uploaded_data


# =====================================================
# GENERATE PREDICTIONS
# =====================================================

def predict_students(data, regressor, classifier):

    X = data[FEATURES]

    predicted_marks = regressor.predict(X)

    predicted_result = classifier.predict(X)

    probabilities = classifier.predict_proba(X)

    # Find probability of passing
    pass_class_position = np.where(
        classifier.classes_ == 1
    )[0]

    if len(pass_class_position) > 0:
        pass_probability = probabilities[
            :, pass_class_position[0]
        ]
    else:
        pass_probability = np.zeros(len(data))

    results = data.copy()

    results["Predicted_Marks"] = np.clip(
        predicted_marks, 0, 100
    ).round(2)

    results["Pass_Probability"] = (
        pass_probability * 100
    ).round(2)

    results["Predicted_Result"] = [
        "Pass" if result == 1 else "Fail"
        for result in predicted_result
    ]

    results["Status"] = [
        "Needs Attention"
        if result == 0 or marks < PASS_MARK or probability < 60
        else "On Track"
        for result, marks, probability in zip(
            predicted_result,
            results["Predicted_Marks"],
            results["Pass_Probability"]
        )
    ]

    return results


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

def show_importance_heatmap(regressor, classifier):

    importance_data = pd.DataFrame({
        "Feature": FEATURES,
        "Marks Prediction": regressor.feature_importances_,
        "Pass/Fail Prediction": classifier.feature_importances_
    })

    heatmap_data = importance_data.set_index("Feature").T

    fig = px.imshow(
        heatmap_data,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="Greens",
        labels={
            "x": "Student Factors",
            "y": "Model",
            "color": "Importance"
        },
        title="Machine Learning Feature Importance"
    )

    fig.update_layout(
        height=300,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This heatmap shows how important each factor was to the "
        "trained models overall. It does not show the exact reason "
        "for an individual student's prediction."
    )


# =====================================================
# PERSONALIZED SUGGESTIONS
# =====================================================

def generate_suggestions(student):

    suggestions = []

    if student["Attendance"] < 75:
        suggestions.append(
            "📅 Improve attendance. Try to maintain at least 75% attendance."
        )

    if student["Assignment_Score"] < 50:
        suggestions.append(
            "📝 Focus on assignments. Complete pending work and practise regularly."
        )

    if student["Internal_Marks"] < 50:
        suggestions.append(
            "📚 Improve internal assessment performance through revision and practice tests."
        )

    if student["Previous_Marks"] < 50:
        suggestions.append(
            "🎯 Revise previous topics and identify subjects where you lost marks."
        )

    if student["Study_Hours"] < 2:
        suggestions.append(
            "⏰ Try studying consistently for at least 2 hours every day."
        )

    elif student["Study_Hours"] >= 6:
        suggestions.append(
            "🌿 Maintain a healthy study routine with sufficient breaks and sleep."
        )

    if student["Pass_Probability"] < 60:
        suggestions.append(
            "⚠️ Your predicted pass probability is low. Prioritize weaker subjects and seek guidance."
        )

    if not suggestions:
        suggestions.append(
            "🌟 Good work! Maintain your current academic routine and continue improving."
        )

    return suggestions


# =====================================================
# INDIVIDUAL STUDENT DASHBOARD
# =====================================================

def show_individual_dashboard(
    student,
    training_data,
    regressor,
    classifier
):

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">👨‍🎓 Individual Student Dashboard</div>',
        unsafe_allow_html=True
    )

    st.write("Detailed academic profile and prediction analysis.")

    # Student information
    st.markdown(
        f"""
        <div class="student-card">
            <div class="student-name">
                👤 {student['Student_ID']}
            </div>
            <div class="student-info">
                Academic Performance Profile
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Prediction metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Final Marks",
        f"{student['Predicted_Marks']}/100"
    )

    col2.metric(
        "Pass Probability",
        f"{student['Pass_Probability']}%"
    )

    col3.metric(
        "Predicted Result",
        student["Predicted_Result"]
    )

    if student["Status"] == "On Track":
        st.success("🌟 This student is currently predicted to be On Track.")
    else:
        st.warning("⚠️ This student may need additional academic support.")

    # -------------------------------------------------
    # STUDENT INPUT VALUES
    # -------------------------------------------------

    st.markdown("### 📊 Student Performance Overview")

    metric_columns = st.columns(5)

    for i, feature in enumerate(FEATURES):

        value = student[feature]

        if feature == "Study_Hours":
            display_value = f"{value:g} hrs"
        else:
            display_value = f"{value:g}/100"

        metric_columns[i].metric(
            feature.replace("_", " "),
            display_value
        )

    # -------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------

    st.markdown("### 📈 Individual Performance Visualization")

    # Compare student with training dataset averages
    comparison = pd.DataFrame({
        "Factor": FEATURES,
        "Student": [
            student[feature] for feature in FEATURES
        ],
        "Dataset Average": [
            training_data[feature].mean()
            for feature in FEATURES
        ]
    })

    comparison_long = comparison.melt(
        id_vars="Factor",
        var_name="Category",
        value_name="Value"
    )

    fig = px.bar(
        comparison_long,
        x="Factor",
        y="Value",
        color="Category",
        barmode="group",
        title="Student Performance vs Training Dataset Average",
        color_discrete_sequence=["#6DAD83", "#A7B5D6"]
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # PERCENTILE PROFILE
    # -------------------------------------------------

    st.markdown("### 📍 Student Factor Percentiles")

    percentile_values = []

    for feature in FEATURES:

        percentile = (
            training_data[feature] <= student[feature]
        ).mean() * 100

        percentile_values.append(round(percentile, 1))

    percentile_data = pd.DataFrame({
        "Factor": FEATURES,
        "Percentile": percentile_values
    })

    fig = px.bar(
        percentile_data,
        x="Percentile",
        y="Factor",
        orientation="h",
        range_x=[0, 100],
        text="Percentile",
        title="How This Student Compares with the Training Dataset",
        color="Percentile",
        color_continuous_scale="Greens"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Percentile indicates the percentage of training records "
        "with a value equal to or lower than this student's value."
    )

    # -------------------------------------------------
    # HEATMAP
    # -------------------------------------------------

    st.markdown("### 🔥 Prediction Feature Importance Heatmap")

    show_importance_heatmap(regressor, classifier)

    # -------------------------------------------------
    # SUGGESTIONS
    # -------------------------------------------------

    st.markdown("### 💡 Personalized Academic Suggestions")

    suggestions = generate_suggestions(student)

    for suggestion in suggestions:
        st.info(suggestion)

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    st.markdown("### 📝 Student Summary")

    if student["Status"] == "On Track":

        st.success(
            "This student is predicted to be on track. "
            "Continue maintaining consistent academic performance."
        )

    else:

        st.warning(
            "This student may require additional academic attention. "
            "Use the suggestions above to identify areas for improvement."
        )


# =====================================================
# MAIN WEBSITE HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🎓 Student Performance Insights</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Understand student performance, predict academic outcomes and discover opportunities for improvement.</div>',
    unsafe_allow_html=True
)

st.write("")


# =====================================================
# LOAD TRAINING DATA AND MODELS
# =====================================================

try:

    training_data = load_training_data()

    if training_data is None:

        st.error(
            "Training dataset not found. Please keep "
            "'student_performance.csv' in the same folder as app.py."
        )

        st.stop()

    regressor, classifier = train_models(training_data)

except Exception as error:

    st.error(f"Error loading training data: {error}")
    st.stop()


# =====================================================
# CSV UPLOAD SECTION
# =====================================================

st.markdown("## 📁 Upload Student Dataset")

st.write(
    "Upload a CSV file containing student attendance, assignment "
    "scores, internal marks, previous marks and study hours."
)

uploaded_file = st.file_uploader(
    "Choose your student CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    file_signature = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    # Clear old predictions if a different CSV is uploaded
    if st.session_state.get("file_signature") != file_signature:

        st.session_state.pop("student_results", None)

        st.session_state["file_signature"] = file_signature

    try:

        uploaded_data = pd.read_csv(uploaded_file)

        st.success(
            f"CSV uploaded successfully! {len(uploaded_data)} records found."
        )

        with st.expander("View uploaded CSV preview"):

            st.dataframe(
                uploaded_data.head(10),
                use_container_width=True
            )

        st.write("")

        if st.button(
            "🚀 Predict All Students",
            type="primary",
            use_container_width=True
        ):

            if len(uploaded_data) == 0:

                st.error("Your CSV file is empty.")

            else:

                cleaned_data = prepare_uploaded_data(
                    uploaded_data,
                    training_data
                )

                results = predict_students(
                    cleaned_data,
                    regressor,
                    classifier
                )

                st.session_state["student_results"] = results

                st.success("Predictions generated successfully!")

    except Exception as error:

        st.error(f"Unable to process CSV: {error}")


# =====================================================
# DISPLAY MAIN DASHBOARD
# =====================================================

if "student_results" in st.session_state:

    results = st.session_state["student_results"]

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">📊 Student Analytics Dashboard</div>',
        unsafe_allow_html=True
    )

    # -------------------------------------------------
    # MAIN METRICS
    # -------------------------------------------------

    total_students = len(results)

    passed_students = (
        results["Predicted_Result"] == "Pass"
    ).sum()

    failed_students = (
        results["Predicted_Result"] == "Fail"
    ).sum()

    at_risk_students = (
        results["Status"] == "Needs Attention"
    ).sum()

    average_marks = results["Predicted_Marks"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👨‍🎓 Total Students",
        total_students
    )

    col2.metric(
        "✅ Predicted Pass",
        int(passed_students)
    )

    col3.metric(
        "⚠️ Needs Attention",
        int(at_risk_students)
    )

    col4.metric(
        "📈 Average Predicted Marks",
        f"{average_marks:.2f}"
    )

    # -------------------------------------------------
    # MAIN CHARTS
    # -------------------------------------------------

    st.markdown("### 📊 Overall Student Performance")

    chart1, chart2 = st.columns(2)

    with chart1:

        result_counts = results[
            "Predicted_Result"
        ].value_counts().reset_index()

        result_counts.columns = [
            "Result",
            "Students"
        ]

        fig = px.pie(
            result_counts,
            names="Result",
            values="Students",
            hole=0.55,
            title="Predicted Pass and Fail Distribution",
            color="Result",
            color_discrete_map={
                "Pass": "#65A982",
                "Fail": "#E27C7C"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with chart2:

        fig = px.histogram(
            results,
            x="Predicted_Marks",
            nbins=15,
            title="Predicted Marks Distribution",
            color_discrete_sequence=["#83B99A"]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------------------------------
    # STUDENT RESULTS TABLE
    # -------------------------------------------------

    st.markdown("### 📋 Individual Student Prediction Results")

    display_columns = [
        "Student_ID",
        "Attendance",
        "Assignment_Score",
        "Internal_Marks",
        "Previous_Marks",
        "Study_Hours",
        "Predicted_Marks",
        "Pass_Probability",
        "Predicted_Result",
        "Status"
    ]

    st.dataframe(
        results[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------------------------
    # DOWNLOAD RESULTS
    # -------------------------------------------------

    csv_output = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Student Prediction Results",
        data=csv_output,
        file_name="student_prediction_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    # -------------------------------------------------
    # INDIVIDUAL STUDENT CARDS
    # -------------------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">👥 Individual Student Profiles</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Choose a student below to explore their individual academic dashboard."
    )

    # Search students
    search_text = st.text_input(
        "🔍 Search Student ID",
        placeholder="Enter student ID..."
    )

    filtered_results = results.copy()

    if search_text:

        filtered_results = filtered_results[
            filtered_results["Student_ID"]
            .astype(str)
            .str.contains(search_text, case=False, na=False)
        ]

    if len(filtered_results) == 0:

        st.warning("No matching students found.")

    else:

        st.write(
            f"Showing {len(filtered_results)} student(s)."
        )

        # Display student cards in three columns
        student_indices = filtered_results.index.tolist()

        for start in range(0, len(student_indices), 3):

            columns = st.columns(3)

            for column, student_index in zip(
                columns,
                student_indices[start:start + 3]
            ):

                student = results.loc[student_index]

                with column:

                    status_color = (
                        "#218653"
                        if student["Status"] == "On Track"
                        else "#D14D4D"
                    )

                    st.markdown(
                        f"""
                        <div class="student-card">
                            <div style="font-size:30px;">👨‍🎓</div>
                            <div class="student-name">
                                {student['Student_ID']}
                            </div>
                            <p>Predicted Marks:
                                <b>{student['Predicted_Marks']}</b>
                            </p>
                            <p>Pass Probability:
                                <b>{student['Pass_Probability']}%</b>
                            </p>
                            <p style="color:{status_color};font-weight:bold;">
                                {student['Status']}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # -------------------------------------------------
    # SELECT INDIVIDUAL STUDENT
    # -------------------------------------------------

    st.markdown("---")

    st.markdown("### 🔎 Explore Individual Student Dashboard")

    student_options = results.index.tolist()

    selected_index = st.selectbox(
        "Select a student",
        options=student_options,
        format_func=lambda i: (
            f"{results.loc[i, 'Student_ID']} — "
            f"{results.loc[i, 'Predicted_Result']} — "
            f"{results.loc[i, 'Predicted_Marks']} marks"
        )
    )

    selected_student = results.loc[selected_index]

    # Display selected student's full dashboard
    show_individual_dashboard(
        selected_student,
        training_data,
        regressor,
        classifier
    )

else:

    # -------------------------------------------------
    # BEFORE CSV UPLOAD
    # -------------------------------------------------

    st.markdown("---")

    st.markdown("## 💡 How to Use This Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 📁 Step 1")

        st.write(
            "Upload your student CSV file containing academic information."
        )

    with col2:

        st.markdown("### 🤖 Step 2")

        st.write(
            "The machine learning models predict final marks and pass probability."
        )

    with col3:

        st.markdown("### 📊 Step 3")

        st.write(
            "Explore individual student dashboards, visualizations, heatmaps and suggestions."
        )

    st.info(
        "Upload a CSV file above to start analysing student performance."
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#89968D;font-size:13px;">
        Student Performance Insights | Data Science Project
        <br>
        Machine Learning • Data Visualization • Academic Analytics
    </div>
    """,
    unsafe_allow_html=True
)
