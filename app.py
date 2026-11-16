
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
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #E3EAE3;
    margin-bottom: 10px;
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
# INDIAN STUDENT NAME PROFILES
# Gender is explicitly assigned with each generated name.
# No gender is inferred from a name.
# =====================================================

MALE_NAMES = [
    "Aarav Sharma", "Arjun Reddy", "Rohan Kumar",
    "Aditya Shetty", "Vikram Rao", "Karthik Gowda",
    "Rahul Verma", "Sanjay Patil", "Ishaan Mehta",
    "Varun Nair", "Pranav Joshi", "Abhishek Das",
    "Manish Yadav", "Suraj Kulkarni", "Nikhil Bhat"
]

FEMALE_NAMES = [
    "Ananya Sharma", "Priya Reddy", "Diya Kumar",
    "Sneha Shetty", "Aishwarya Rao", "Kavya Gowda",
    "Meera Verma", "Pooja Patil", "Isha Mehta",
    "Nandini Nair", "Sanjana Joshi", "Swathi Das",
    "Neha Yadav", "Riya Kulkarni", "Shreya Bhat"
]


def make_profile(identifier):
    """Generate a stable example Indian student identity."""

    value = int(
        hashlib.md5(str(identifier).encode()).hexdigest(),
        16
    )

    if value % 2 == 0:
        gender = "Male"
        name = MALE_NAMES[(value // 2) % len(MALE_NAMES)]
        avatar = "👨‍🎓"
    else:
        gender = "Female"
        name = FEMALE_NAMES[(value // 2) % len(FEMALE_NAMES)]
        avatar = "👩‍🎓"

    return name, gender, avatar


# =====================================================
# LOAD TRAINING DATA
# =====================================================

@st.cache_data
def load_training_data():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "student_performance.csv was not found beside app.py."
        )

    data = pd.read_csv(DATASET_PATH)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Training CSV is missing: " + ", ".join(missing)
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
# CLEAN UPLOADED DATA
# =====================================================

def clean_student_data(data, training_data):

    data = data.copy()

    missing = [
        col for col in FEATURES
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    if "Student_ID" not in data.columns:
        data["Student_ID"] = [
            f"STU{i+1:03d}" for i in range(len(data))
        ]

    if "Student_Name" not in data.columns:
        data["Student_Name"] = ""

    if "Gender" not in data.columns:
        data["Gender"] = ""

    data["Student_ID"] = data["Student_ID"].astype(str)

    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(training_data[col].median())

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
# PREDICTION FUNCTION
# =====================================================

def predict_students(data, regressor, classifier):

    X = data[FEATURES]

    marks = np.clip(regressor.predict(X), 0, 100)

    result = classifier.predict(X)

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
        for value in result
    ]

    output["Status"] = [
        "Needs Attention"
        if value == 0 or mark < PASS_MARK or probability < 60
        else "On Track"
        for value, mark, probability in zip(
            result, marks, pass_probability
        )
    ]

    # Assign names and avatars where not supplied
    names = []
    genders = []
    avatars = []

    for _, student in output.iterrows():

        name = str(student["Student_Name"]).strip()
        gender = str(student["Gender"]).strip().title()

        if not name or name.lower() == "nan":
            name, gender, avatar = make_profile(
                student["Student_ID"]
            )
        else:
            if gender == "Male":
                avatar = "👨‍🎓"
            elif gender == "Female":
                avatar = "👩‍🎓"
            else:
                # If a supplied name has no gender selected,
                # use a neutral avatar rather than guessing.
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
# ADD STUDENTS TO SESSION
# =====================================================

def add_students(new_results):

    if "student_results" not in st.session_state:
        st.session_state.student_results = pd.DataFrame()

    existing = st.session_state.student_results.copy()

    new_results = new_results.copy()

    next_id = (
        int(existing["Record_ID"].max()) + 1
        if not existing.empty else 1
    )

    new_results["Record_ID"] = range(
        next_id, next_id + len(new_results)
    )

    if existing.empty:
        combined = new_results
    else:
        combined = pd.concat(
            [existing, new_results],
            ignore_index=True
        )

    st.session_state.student_results = combined


# =====================================================
# SUGGESTIONS
# =====================================================

def get_suggestions(student):

    suggestions = []

    if student["Attendance"] < 75:
        suggestions.append(
            "📅 Improve attendance and aim for at least 75%."
        )

    if student["Assignment_Score"] < 50:
        suggestions.append(
            "📝 Spend more time completing assignments and practising questions."
        )

    if student["Internal_Marks"] < 50:
        suggestions.append(
            "📚 Revise internal assessment topics and take practice tests."
        )

    if student["Previous_Marks"] < 50:
        suggestions.append(
            "🎯 Revise previously difficult topics and strengthen fundamentals."
        )

    if student["Study_Hours"] < 2:
        suggestions.append(
            "⏰ Build a regular study routine of at least 2 hours per day."
        )

    if student["Pass_Probability"] < 60:
        suggestions.append(
            "⚠️ Arrange additional academic support and monitor progress."
        )

    if not suggestions:
        suggestions.append(
            "🌟 Good work! Maintain your current routine and keep improving."
        )

    return suggestions


# =====================================================
# INDIVIDUAL DASHBOARD PAGE
# =====================================================

def show_student_page(student, training_data, regressor, classifier):

    if st.button("← Back to Main Dashboard"):
        st.query_params.clear()
        st.rerun()

    st.markdown("---")

    st.markdown(
        f"""
        <div class="student-card">
            <div style="font-size:65px;">{student['Avatar']}</div>
            <h1 style="color:#294C3A;">{student['Student_Name']}</h1>
            <p>Student ID: {student['Student_ID']}</p>
            <p>Gender: {student['Gender']}</p>
            <p>Individual Academic Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Student prediction metrics
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

    if student["Status"] == "On Track":
        st.success("This student is predicted to be On Track.")
    else:
        st.warning("This student may need additional academic support.")

    # Academic input metrics
    st.markdown("## 📊 Academic Information")

    cols = st.columns(5)

    for i, feature in enumerate(FEATURES):

        value = student[feature]

        label = feature.replace("_", " ")

        if feature == "Study_Hours":
            value_text = f"{value:g} hrs"
        else:
            value_text = f"{value:g}/100"

        cols[i].metric(label, value_text)

    # Student vs training average
    st.markdown("## 📈 Individual Performance Visualization")

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

    chart_data = comparison.melt(
        id_vars="Factor",
        var_name="Category",
        value_name="Value"
    )

    fig = px.bar(
        chart_data,
        x="Factor",
        y="Value",
        color="Category",
        barmode="group",
        color_discrete_sequence=["#6EAE84", "#A7B8D5"],
        title="Student Values Compared with Training Dataset Average"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Percentile chart
    st.markdown("## 📍 Student Performance Percentiles")

    percentiles = []

    for feature in FEATURES:
        percentile = (
            training_data[feature] <= student[feature]
        ).mean() * 100

        percentiles.append(round(percentile, 1))

    percentile_data = pd.DataFrame({
        "Factor": FEATURES,
        "Percentile": percentiles
    })

    fig = px.bar(
        percentile_data,
        x="Percentile",
        y="Factor",
        orientation="h",
        range_x=[0, 100],
        text="Percentile",
        color="Percentile",
        color_continuous_scale="Greens",
        title="Comparison with Training Dataset"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Feature importance heatmap
    st.markdown("## 🔥 Prediction Heatmap")

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Marks Model": regressor.feature_importances_,
        "Pass/Fail Model": classifier.feature_importances_
    }).set_index("Feature").T

    fig = px.imshow(
        importance,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="Greens",
        title="Model Feature Importance"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This is the overall importance of factors to the trained models, "
        "not an exact explanation of this student's individual prediction."
    )

    # Suggestions
    st.markdown("## 💡 Personalized Academic Suggestions")

    for suggestion in get_suggestions(student):
        st.info(suggestion)

    # Summary
    st.markdown("## 📝 Performance Summary")

    if student["Status"] == "On Track":
        st.success(
            "This student is predicted to be on track. "
            "Continue maintaining consistent academic performance."
        )
    else:
        st.warning(
            "This student may need additional support. "
            "Focus on the weaker academic factors listed above."
        )


# =====================================================
# INITIALIZE SESSION
# =====================================================

if "student_results" not in st.session_state:
    st.session_state.student_results = pd.DataFrame()

if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None


# =====================================================
# LOAD TRAINING DATA
# =====================================================

try:
    training_data = load_training_data()
    regressor, classifier = train_models(training_data)

except Exception as error:
    st.error(f"Training error: {error}")
    st.stop()


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🎓 Student Performance Insights</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Academic analytics • Machine learning • Student performance prediction</div>',
    unsafe_allow_html=True
)


# =====================================================
# ROUTE TO INDIVIDUAL STUDENT DASHBOARD
# =====================================================

selected_record = st.query_params.get("student")

if selected_record is not None:

    all_results = st.session_state.student_results

    if not all_results.empty:

        match = all_results[
            all_results["Record_ID"].astype(str) == str(selected_record)
        ]

        if not match.empty:

            selected_student = match.iloc[0]

            show_student_page(
                selected_student,
                training_data,
                regressor,
                classifier
            )

            st.stop()

    st.warning("Student dashboard not found.")
    if st.button("Return to Main Dashboard"):
        st.query_params.clear()
        st.rerun()

    st.stop()


# =====================================================
# MAIN DASHBOARD
# =====================================================

st.markdown("---")

# -----------------------------------------------------
# BULK CSV UPLOAD
# -----------------------------------------------------

st.markdown("## 📁 Bulk Student Upload")

st.write(
    "Upload a CSV containing multiple students to predict their academic performance."
)

uploaded_file = st.file_uploader(
    "Choose student CSV",
    type=["csv"],
    key="bulk_csv"
)

if uploaded_file is not None:

    try:

        signature = hashlib.md5(
            uploaded_file.getvalue()
        ).hexdigest()

        uploaded_data = pd.read_csv(uploaded_file)

        st.success(
            f"{len(uploaded_data)} student records detected."
        )

        with st.expander("Preview uploaded CSV"):
            st.dataframe(
                uploaded_data.head(10),
                use_container_width=True
            )

        if st.button(
            "🚀 Predict All Students",
            type="primary",
            key="bulk_predict"
        ):

            if signature != st.session_state.upload_signature:

                # Remove old bulk predictions only when a new CSV is used.
                old = st.session_state.student_results

                if not old.empty:
                    old = old[old["Source"] != "Bulk CSV"]

                st.session_state.student_results = old
                st.session_state.upload_signature = signature

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

            st.success("Bulk predictions completed!")

    except Exception as error:
        st.error(f"CSV error: {error}")


# -----------------------------------------------------
# SINGLE STUDENT INPUT
# -----------------------------------------------------

st.markdown("---")

st.markdown("## 👤 Single Student Prediction")

st.write(
    "Enter one student's details manually without uploading a CSV."
)

with st.form("single_student_form"):

    name = st.text_input(
        "Student Name (optional)",
        placeholder="Leave blank for an automatically assigned Indian name"
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Not specified"]
    )

    student_id = st.text_input(
        "Student ID (optional)",
        placeholder="Example: STU101"
    )

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0
        )

        assignment = st.number_input(
            "Assignment Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        internal = st.number_input(
            "Internal Marks",
            min_value=0.0,
            max_value=100.0,
            value=65.0
        )

    with col2:
        previous = st.number_input(
            "Previous Marks",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        study_hours = st.number_input(
            "Study Hours Per Day",
            min_value=0.0,
            max_value=24.0,
            value=3.0
        )

    submitted = st.form_submit_button(
        "🔮 Predict Individual Student",
        use_container_width=True
    )

if submitted:

    if not student_id.strip():
        student_id = f"SINGLE{len(st.session_state.student_results) + 1:03d}"

    if not name.strip():
        generated_name, generated_gender, avatar = make_profile(student_id)

        name = generated_name
        gender = generated_gender

    else:
        if gender == "Male":
            avatar = "👨‍🎓"
        elif gender == "Female":
            avatar = "👩‍🎓"
        else:
            avatar = "🧑‍🎓"

    single_data = pd.DataFrame([{
        "Student_ID": student_id,
        "Student_Name": name,
        "Gender": gender,
        "Attendance": attendance,
        "Assignment_Score": assignment,
        "Internal_Marks": internal,
        "Previous_Marks": previous,
        "Study_Hours": study_hours
    }])

    single_prediction = predict_students(
        single_data,
        regressor,
        classifier
    )

    single_prediction["Source"] = "Single Entry"

    add_students(single_prediction)

    st.success(
        f"Prediction generated for {name}. "
        "Open their personal dashboard below."
    )


# =====================================================
# DISPLAY MAIN RESULTS
# =====================================================

results = st.session_state.student_results

if not results.empty:

    st.markdown("---")

    st.markdown("## 📊 Main Student Analytics Dashboard")

    total = len(results)

    passed = (
        results["Predicted_Result"] == "Pass"
    ).sum()

    failed = (
        results["Predicted_Result"] == "Fail"
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

    # Main charts
    st.markdown("### Overall Prediction Analysis")

    left, right = st.columns(2)

    with left:

        counts = results["Predicted_Result"].value_counts()

        fig = px.pie(
            names=counts.index,
            values=counts.values,
            hole=0.55,
            title="Pass and Fail Distribution",
            color=counts.index,
            color_discrete_map={
                "Pass": "#65A982",
                "Fail": "#E27C7C"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        fig = px.histogram(
            results,
            x="Predicted_Marks",
            nbins=15,
            title="Predicted Marks Distribution",
            color_discrete_sequence=["#83B99A"]
        )

        st.plotly_chart(fig, use_container_width=True)

    # Results table
    st.markdown("### 📋 Prediction Results")

    display_columns = [
        "Student_ID",
        "Student_Name",
        "Gender",
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

    # Download
    st.download_button(
        "⬇️ Download All Prediction Results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="student_prediction_results.csv",
        mime="text/csv"
    )

    # -------------------------------------------------
    # STUDENT CARDS
    # -------------------------------------------------

    st.markdown("---")

    st.markdown("## 👥 Individual Student Profiles")

    st.write(
        "Click the student's avatar button to open their personal dashboard."
    )

    search = st.text_input(
        "🔍 Search student by name or ID"
    )

    filtered = results.copy()

    if search:
        filtered = filtered[
            filtered["Student_Name"].astype(str).str.contains(
                search, case=False, na=False
            )
            |
            filtered["Student_ID"].astype(str).str.contains(
                search, case=False, na=False
            )
        ]

    if filtered.empty:
        st.info("No students found.")
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
                            <div style="font-size:45px;">
                                {student['Avatar']}
                            </div>
                            <h3 style="color:#294C3A;">
                                {student['Student_Name']}
                            </h3>
                            <p>Student ID: {student['Student_ID']}</p>
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

                    # Clicking the avatar button opens a separate dashboard view
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
        "Upload a CSV or enter a single student's details to begin."
    )

    st.markdown("### How it works")

    a, b, c = st.columns(3)

    a.markdown(
        "**📁 Bulk Upload**\n\n"
        "Upload several student records at once."
    )

    b.markdown(
        "**👤 Single Entry**\n\n"
        "Enter one student's academic details manually."
    )

    c.markdown(
        "**📊 Personal Dashboard**\n\n"
        "Open individual student profiles with predictions, charts and suggestions."
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#89968D;">
        Student Performance Insights | Data Science Project
        <br>
        Machine Learning • Data Analytics • Academic Insights
    </div>
    """,
    unsafe_allow_html=True
)

