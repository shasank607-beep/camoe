
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import hashlib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Student Performance Insights",
    page_icon="🎓",
    layout="wide"
)

PASS_MARK = 40

FEATURES = [
    "Attendance",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Marks",
    "Study_Hours"
]

REQUIRED_COLUMNS = FEATURES + ["Final_Marks"]

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "student_performance.csv"
)

# =====================================================
# THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #F6F8F5;
    color: #26352D;
}
[data-testid="stSidebar"] {
    background: #EAF0E9;
}
.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #294C3A;
}
.subtitle {
    color: #718078;
    font-size: 15px;
}
.student-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #E3EAE3;
    text-align: center;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 14px;
    border: 1px solid #E3EAE3;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# SAMPLE INDIAN STUDENT PROFILES
# =====================================================

MALE_NAMES = [
    "Aarav Sharma", "Arjun Reddy", "Rohan Kumar",
    "Aditya Shetty", "Vikram Rao", "Karthik Gowda",
    "Rahul Verma", "Sanjay Patil", "Ishaan Mehta",
    "Varun Nair", "Pranav Joshi", "Abhishek Das"
]

FEMALE_NAMES = [
    "Ananya Sharma", "Priya Reddy", "Diya Kumar",
    "Sneha Shetty", "Aishwarya Rao", "Kavya Gowda",
    "Meera Verma", "Pooja Patil", "Isha Mehta",
    "Nandini Nair", "Sanjana Joshi", "Swathi Das"
]


def generate_profile(identifier):

    value = int(
        hashlib.md5(str(identifier).encode()).hexdigest(),
        16
    )

    if value % 2 == 0:
        return (
            MALE_NAMES[(value // 2) % len(MALE_NAMES)],
            "Male",
            "👨‍🎓"
        )

    return (
        FEMALE_NAMES[(value // 2) % len(FEMALE_NAMES)],
        "Female",
        "👩‍🎓"
    )


# =====================================================
# SESSION STATE
# =====================================================

if "students" not in st.session_state:
    st.session_state.students = pd.DataFrame()

if "next_record_id" not in st.session_state:
    st.session_state.next_record_id = 1

if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = set()


# =====================================================
# LOAD TRAINING DATA
# =====================================================

@st.cache_data
def load_training_data():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "student_performance.csv not found beside app.py"
        )

    data = pd.read_csv(DATASET_PATH)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    data = data[REQUIRED_COLUMNS].copy()

    for col in REQUIRED_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Final_Marks"])

    if len(data) < 2:
        raise ValueError("At least two valid training records are required.")

    for col in FEATURES:
        median = data[col].median()
        if pd.isna(median):
            median = 0
        data[col] = data[col].fillna(median)

    for col in [
        "Attendance",
        "Assignment_Score",
        "Internal_Marks",
        "Previous_Marks",
        "Final_Marks"
    ]:
        data[col] = data[col].clip(0, 100)

    data["Study_Hours"] = data["Study_Hours"].clip(0, 24)

    return data


# =====================================================
# TRAIN MODELS
# =====================================================

@st.cache_resource
def train_models(data):

    X = data[FEATURES]

    y_marks = data["Final_Marks"]

    y_result = (y_marks >= PASS_MARK).astype(int)

    regressor = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    classifier = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    regressor.fit(X, y_marks)
    classifier.fit(X, y_result)

    return regressor, classifier


# =====================================================
# CLEAN INPUT DATA
# =====================================================

def clean_student_data(data, training_data):

    data = data.copy()

    missing = [
        col for col in FEATURES
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    if "Student_ID" not in data.columns:
        data["Student_ID"] = [
            f"STU{i+1:03d}" for i in range(len(data))
        ]

    if "Student_Name" not in data.columns:
        data["Student_Name"] = ""

    if "Gender" not in data.columns:
        data["Gender"] = ""

    for col in ["Student_ID", "Student_Name", "Gender"]:
        data[col] = data[col].fillna("").astype(str)

    for col in FEATURES:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

        data[col] = data[col].fillna(
            training_data[col].median()
        )

    for col in [
        "Attendance",
        "Assignment_Score",
        "Internal_Marks",
        "Previous_Marks"
    ]:
        data[col] = data[col].clip(0, 100)

    data["Study_Hours"] = data["Study_Hours"].clip(0, 24)

    return data


# =====================================================
# PREDICTION
# =====================================================

def predict_students(data, regressor, classifier):

    X = data[FEATURES]

    marks = np.clip(regressor.predict(X), 0, 100)

    predicted_result = classifier.predict(X)

    probabilities = classifier.predict_proba(X)

    pass_positions = np.where(classifier.classes_ == 1)[0]

    if len(pass_positions):
        pass_probability = probabilities[:, pass_positions[0]] * 100
    else:
        pass_probability = np.zeros(len(data))

    output = data.copy()

    output["Predicted_Marks"] = np.round(marks, 2)
    output["Pass_Probability"] = np.round(pass_probability, 2)

    output["Predicted_Result"] = [
        "Pass" if value == 1 else "Fail"
        for value in predicted_result
    ]

    output["Status"] = [
        "Needs Attention"
        if result == 0 or mark < PASS_MARK or probability < 60
        else "On Track"
        for result, mark, probability in zip(
            predicted_result, marks, pass_probability
        )
    ]

    names = []
    genders = []
    avatars = []

    for _, student in output.iterrows():

        name = student["Student_Name"].strip()
        gender = student["Gender"].strip().title()

        if not name:
            name, gender, avatar = generate_profile(
                student["Student_ID"]
            )
        elif gender == "Male":
            avatar = "👨‍🎓"
        elif gender == "Female":
            avatar = "👩‍🎓"
        else:
            gender = "Not specified"
            avatar = "🧑‍🎓"

        names.append(name)
        genders.append(gender)
        avatars.append(avatar)

    output["Student_Name"] = names
    output["Gender"] = genders
    output["Avatar"] = avatars

    return output


# =====================================================
# ADD STUDENTS
# =====================================================

def add_students(new_data):

    new_data = new_data.copy()

    start = st.session_state.next_record_id

    new_data["Record_ID"] = range(
        start,
        start + len(new_data)
    )

    st.session_state.next_record_id += len(new_data)

    if st.session_state.students.empty:
        st.session_state.students = new_data
    else:
        st.session_state.students = pd.concat(
            [st.session_state.students, new_data],
            ignore_index=True
        )


# =====================================================
# PERSONALIZED SUGGESTIONS
# =====================================================

def get_suggestions(student):

    suggestions = []

    if student["Attendance"] < 75:
        suggestions.append(
            "📅 Improve attendance and aim for at least 75%."
        )

    if student["Assignment_Score"] < 50:
        suggestions.append(
            "📝 Complete pending assignments and practise regularly."
        )

    if student["Internal_Marks"] < 50:
        suggestions.append(
            "📚 Revise internal assessment topics and take practice tests."
        )

    if student["Previous_Marks"] < 50:
        suggestions.append(
            "🎯 Strengthen fundamentals by revising previous topics."
        )

    if student["Study_Hours"] < 2:
        suggestions.append(
            "⏰ Build a consistent daily study routine."
        )

    if student["Pass_Probability"] < 60:
        suggestions.append(
            "⚠️ Additional academic support may be helpful."
        )

    if not suggestions:
        suggestions.append(
            "🌟 Great work! Maintain your current academic routine."
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

    # BACK BUTTON

    if st.button("← Back to Main Dashboard"):
        st.query_params.clear()
        st.rerun()

    st.markdown("---")

    # PROFILE

    st.markdown(
        f"""
        <div class="student-card">
            <div style="font-size:75px;">{student['Avatar']}</div>
            <h1 style="color:#294C3A;">{student['Student_Name']}</h1>
            <p>Student ID: {student['Student_ID']}</p>
            <p>Gender: {student['Gender']}</p>
            <p>Personal Academic Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =================================================
    # PREDICTION METRICS
    # =================================================

    st.markdown("## 🎯 Prediction Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Predicted Final Marks",
        f"{student['Predicted_Marks']}/100"
    )

    c2.metric(
        "Pass Probability",
        f"{student['Pass_Probability']}%"
    )

    c3.metric(
        "Predicted Result",
        student["Predicted_Result"]
    )

    # =================================================
    # ATTENDANCE PIE CHART
    # =================================================

    st.markdown("---")
    st.markdown("## 📅 Attendance Analysis")

    attendance = float(student["Attendance"])

    attendance_data = pd.DataFrame({
        "Category": ["Present", "Remaining"],
        "Percentage": [attendance, 100 - attendance]
    })

    fig = px.pie(
        attendance_data,
        names="Category",
        values="Percentage",
        hole=0.60,
        color="Category",
        color_discrete_map={
            "Present": "#65A982",
            "Remaining": "#E5EAE5"
        },
        title="Attendance Distribution"
    )

    fig.update_traces(textinfo="label+percent")

    fig.update_layout(
        annotations=[
            dict(
                text=f"{attendance:.1f}%",
                x=0.5,
                y=0.5,
                font_size=25,
                showarrow=False
            )
        ],
        paper_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    if attendance < 75:
        st.warning("Attendance is below the 75% reference target.")
    else:
        st.success("Attendance meets the 75% reference target.")

    # =================================================
    # REGRESSION ANALYSIS
    # =================================================

    st.markdown("---")
    st.markdown("## 📈 Attendance vs Academic Performance")

    st.write(
        "The scatter plot shows the relationship between attendance "
        "and actual final marks in the training dataset. The regression "
        "line shows the overall linear trend."
    )

    regression_data = training_data[
        ["Attendance", "Final_Marks"]
    ].copy()

    fig = px.scatter(
        regression_data,
        x="Attendance",
        y="Final_Marks",
        opacity=0.65,
        title="Attendance vs Final Marks",
        labels={
            "Attendance": "Attendance (%)",
            "Final_Marks": "Actual Final Marks"
        },
        trendline="ols",
        trendline_color_override="#E28A55"
    )

    # Highlight selected student's prediction
    fig.add_trace(
        go.Scatter(
            x=[student["Attendance"]],
            y=[student["Predicted_Marks"]],
            mode="markers",
            name="Selected Student Prediction",
            marker=dict(
                size=15,
                color="#315FAD",
                symbol="star",
                line=dict(width=2, color="white")
            )
        )
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The regression line is fitted to the training data. "
        "The star represents this student's predicted marks, "
        "not an observed final result."
    )

    # =================================================
    # RADAR CHART
    # =================================================

    st.markdown("---")
    st.markdown("## 🕸️ Student Performance Radar")

    radar_values = [
        student["Attendance"],
        student["Assignment_Score"],
        student["Internal_Marks"],
        student["Previous_Marks"],
        min(student["Study_Hours"] / 8 * 100, 100)
    ]

    radar_labels = [
        "Attendance",
        "Assignments",
        "Internal Marks",
        "Previous Marks",
        "Study Hours"
    ]

    radar_values_closed = radar_values + [radar_values[0]]
    radar_labels_closed = radar_labels + [radar_labels[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=radar_values_closed,
            theta=radar_labels_closed,
            fill="toself",
            name="Student",
            line_color="#65A982"
        )
    )

    average_values = [
        training_data["Attendance"].mean(),
        training_data["Assignment_Score"].mean(),
        training_data["Internal_Marks"].mean(),
        training_data["Previous_Marks"].mean(),
        min(training_data["Study_Hours"].mean() / 8 * 100, 100)
    ]

    fig.add_trace(
        go.Scatterpolar(
            r=average_values + [average_values[0]],
            theta=radar_labels_closed,
            fill="toself",
            name="Training Average",
            line_color="#A7B8D5",
            opacity=0.65
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title="Student vs Training Average",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # PERFORMANCE COMPARISON
    # =================================================

    st.markdown("---")
    st.markdown("## 📊 Academic Factor Comparison")

    comparison = pd.DataFrame({
        "Factor": FEATURES,
        "Student": [
            student[feature] for feature in FEATURES
        ],
        "Training Average": [
            training_data[feature].mean()
            for feature in FEATURES
        ]
    })

    long_data = comparison.melt(
        id_vars="Factor",
        var_name="Category",
        value_name="Value"
    )

    fig = px.bar(
        long_data,
        x="Factor",
        y="Value",
        color="Category",
        barmode="group",
        title="Student vs Training Dataset Average",
        color_discrete_sequence=[
            "#65A982",
            "#A7B8D5"
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # MARKS DISTRIBUTION
    # =================================================

    st.markdown("---")
    st.markdown("## 📉 Marks Distribution")

    fig = px.histogram(
        training_data,
        x="Final_Marks",
        nbins=15,
        title="Training Dataset Final Marks Distribution",
        color_discrete_sequence=["#83B99A"]
    )

    fig.add_vline(
        x=student["Predicted_Marks"],
        line_dash="dash",
        line_color="#D76D55",
        annotation_text="Student Prediction"
    )

    fig.add_vline(
        x=PASS_MARK,
        line_dash="dot",
        line_color="#315FAD",
        annotation_text="Pass Mark"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # INDIVIDUAL HEATMAP
    # =================================================

    st.markdown("---")
    st.markdown("## 🔥 Individual Criteria Heatmap")

    heatmap_values = {
        "Attendance": student["Attendance"],
        "Assignment": student["Assignment_Score"],
        "Internal Marks": student["Internal_Marks"],
        "Previous Marks": student["Previous_Marks"],
        "Study Hours": min(student["Study_Hours"] / 8 * 100, 100)
    }

    heatmap_data = pd.DataFrame(
        [heatmap_values],
        index=[student["Student_Name"]]
    )

    fig = px.imshow(
        heatmap_data,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=100,
        title="Student Academic Criteria (0–100 scale)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # MODEL FEATURE IMPORTANCE
    # =================================================

    st.markdown("### 🧠 Model Feature Importance")

    importance_data = pd.DataFrame({
        "Feature": FEATURES,
        "Marks Model": regressor.feature_importances_,
        "Pass/Fail Model": classifier.feature_importances_
    }).set_index("Feature").T

    fig = px.imshow(
        importance_data,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="Greens",
        title="Overall Prediction Model Criteria"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This shows overall model feature importance, not the "
        "exact contribution of each factor to this student's prediction."
    )

    # =================================================
    # SUGGESTIONS
    # =================================================

    st.markdown("---")
    st.markdown("## 💡 Personalized Academic Suggestions")

    for suggestion in get_suggestions(student):
        st.info(suggestion)

    # =================================================
    # SUMMARY
    # =================================================

    st.markdown("---")
    st.markdown("## 📝 Student Summary")

    if student["Status"] == "On Track":
        st.success(
            f"{student['Student_Name']} is predicted to be on track. "
            "Maintain the current academic routine."
        )
    else:
        st.warning(
            f"{student['Student_Name']} may need academic support. "
            "Focus on the weaker criteria and follow the suggestions."
        )


# =========================================================
# LOAD DATA AND TRAIN
# =========================================================

try:
    training_data = load_training_data()
    regressor, classifier = train_models(training_data)

except Exception as error:
    st.error(f"Training Error: {error}")
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎓 Student Performance Insights</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Academic Analytics | Machine Learning | Student Predictions</div>',
    unsafe_allow_html=True
)


# =========================================================
# ROUTE TO INDIVIDUAL PAGE
# =========================================================

selected_record = st.query_params.get("student")

if selected_record is not None:

    students = st.session_state.students

    if not students.empty:

        match = students[
            students["Record_ID"].astype(str) == str(selected_record)
        ]

        if not match.empty:

            student = match.iloc[0]

            show_individual_dashboard(
                student,
                training_data,
                regressor,
                classifier
            )

            st.stop()

    st.warning("Student profile not found.")

    if st.button("Return to Dashboard"):
        st.query_params.clear()
        st.rerun()

    st.stop()


# =========================================================
# MAIN PAGE - BULK UPLOAD
# =========================================================

st.markdown("---")
st.markdown("## 📁 Bulk Student CSV Upload")

uploaded_file = st.file_uploader(
    "Upload your student CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        uploaded_data = pd.read_csv(uploaded_file)

        signature = hashlib.md5(
            uploaded_file.getvalue()
        ).hexdigest()

        st.success(
            f"{len(uploaded_data)} student records detected."
        )

        with st.expander("Preview CSV"):
            st.dataframe(uploaded_data.head(10))

        if st.button(
            "🚀 Predict All Students",
            type="primary"
        ):

            if signature in st.session_state.processed_uploads:
                st.warning("This CSV has already been processed.")
            else:

                cleaned = clean_student_data(
                    uploaded_data,
                    training_data
                )

                predictions = predict_students(
                    cleaned,
                    regressor,
                    classifier
                )

                predictions["Source"] = "Bulk CSV"

                add_students(predictions)

                st.session_state.processed_uploads.add(signature)

                st.success("Predictions completed!")

    except Exception as error:
        st.error(f"CSV Error: {error}")


# =========================================================
# SINGLE STUDENT INPUT
# =========================================================

st.markdown("---")
st.markdown("## 👤 Add Individual Student")

with st.form("single_student_form"):

    student_name = st.text_input(
        "Student Name (optional)"
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Not specified"]
    )

    student_id = st.text_input(
        "Student ID (optional)"
    )

    c1, c2 = st.columns(2)

    with c1:
        attendance = st.number_input(
            "Attendance (%)", 0.0, 100.0, 80.0
        )
        assignment = st.number_input(
            "Assignment Score", 0.0, 100.0, 70.0
        )
        internal = st.number_input(
            "Internal Marks", 0.0, 100.0, 65.0
        )

    with c2:
        previous = st.number_input(
            "Previous Marks", 0.0, 100.0, 70.0
        )
        study_hours = st.number_input(
            "Study Hours Per Day", 0.0, 24.0, 3.0
        )

    submitted = st.form_submit_button(
        "🔮 Predict Individual Student",
        use_container_width=True
    )

if submitted:

    current_id = (
        student_id.strip()
        if student_id.strip()
        else f"SINGLE{st.session_state.next_record_id:03d}"
    )

    existing_ids = (
        st.session_state.students["Student_ID"].astype(str).tolist()
        if not st.session_state.students.empty
        else []
    )

    if current_id in existing_ids:

        st.error("Student ID already exists. Use another ID.")

    else:

        single_data = pd.DataFrame([{
            "Student_ID": current_id,
            "Student_Name": student_name,
            "Gender": gender,
            "Attendance": attendance,
            "Assignment_Score": assignment,
            "Internal_Marks": internal,
            "Previous_Marks": previous,
            "Study_Hours": study_hours
        }])

        prediction = predict_students(
            single_data,
            regressor,
            classifier
        )

        prediction["Source"] = "Single Entry"

        add_students(prediction)

        st.success("Individual prediction completed!")


# =========================================================
# MAIN ANALYTICS
# =========================================================

results = st.session_state.students

if not results.empty:

    st.markdown("---")
    st.markdown("## 📊 Overall Student Analytics")

    total = len(results)

    passed = (
        results["Predicted_Result"] == "Pass"
    ).sum()

    attention = (
        results["Status"] == "Needs Attention"
    ).sum()

    average = results["Predicted_Marks"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("👨‍🎓 Total Students", total)
    c2.metric("✅ Predicted Pass", int(passed))
    c3.metric("⚠️ Needs Attention", int(attention))
    c4.metric("📈 Average Marks", f"{average:.2f}")

    # Overall pass/fail chart

    chart1, chart2 = st.columns(2)

    with chart1:

        counts = results["Predicted_Result"].value_counts()

        fig = px.pie(
            names=counts.index,
            values=counts.values,
            hole=0.55,
            title="Predicted Pass and Fail",
            color=counts.index,
            color_discrete_map={
                "Pass": "#65A982",
                "Fail": "#E27C7C"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    with chart2:

        fig = px.histogram(
            results,
            x="Predicted_Marks",
            nbins=15,
            title="Predicted Marks Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Download

    st.download_button(
        "⬇️ Download Prediction Results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="student_prediction_results.csv",
        mime="text/csv"
    )

    # =====================================================
    # STUDENT CARDS
    # =====================================================

    st.markdown("---")
    st.markdown("## 👥 Individual Student Profiles")

    search = st.text_input(
        "🔍 Search student name or ID"
    )

    filtered = results.copy()

    if search:

        filtered = filtered[
            filtered["Student_Name"].str.contains(
                search, case=False, na=False
            )
            |
            filtered["Student_ID"].str.contains(
                search, case=False, na=False
            )
        ]

    if filtered.empty:

        st.info("No matching students.")

    else:

        for start in range(0, len(filtered), 3):

            columns = st.columns(3)

            for col, (_, student) in zip(
                columns,
                filtered.iloc[start:start + 3].iterrows()
            ):

                with col:

                    status_color = (
                        "#218653"
                        if student["Status"] == "On Track"
                        else "#D14D4D"
                    )

                    st.markdown(
                        f"""
                        <div class="student-card">
                            <div style="font-size:55px;">
                                {student['Avatar']}
                            </div>
                            <h3>{student['Student_Name']}</h3>
                            <p>ID: {student['Student_ID']}</p>
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

                    if st.button(
                        f"{student['Avatar']} Open Personal Dashboard",
                        key=f"student_{student['Record_ID']}",
                        use_container_width=True
                    ):

                        st.query_params["student"] = str(
                            student["Record_ID"]
                        )

                        st.rerun()

else:

    st.info(
        "Upload a CSV or enter a student's details to begin."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#89968D;">
        Student Performance Insights
        <br>
        Data Science | Machine Learning | Academic Analytics
    </div>
    """,
    unsafe_allow_html=True
)
