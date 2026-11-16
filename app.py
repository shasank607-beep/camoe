import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Student Performance Insights",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Insights")
st.write("Analyze, visualize and predict student academic performance.")

# ---------------- LOAD DATASET ----------------

file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "student_performance.csv"
)

if not os.path.exists(file_path):
    st.error("student_performance.csv not found!")
    st.stop()

df = pd.read_csv(file_path)

st.success("Dataset loaded successfully!")

# ---------------- CHECK COLUMNS ----------------

features = [
    "Attendance",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Marks",
    "Study_Hours"
]

required = features + ["Final_Marks"]

missing = [col for col in required if col not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# ---------------- DATA CLEANING ----------------

st.header("1. Data Cleaning")

df = df.drop_duplicates().copy()

for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Final_Marks"])

for col in features:
    df[col] = df[col].fillna(df[col].median())

# Keep valid mark and attendance ranges
for col in ["Attendance", "Assignment_Score",
            "Internal_Marks", "Previous_Marks", "Final_Marks"]:
    df[col] = df[col].clip(0, 100)

df["Study_Hours"] = df["Study_Hours"].clip(lower=0)

# Pass mark is 40 for this demonstration
df["Result"] = np.where(df["Final_Marks"] >= 40, "Pass", "Fail")

st.write("Cleaned Dataset")
st.dataframe(df, use_container_width=True)

# ---------------- DASHBOARD ----------------

st.header("2. Performance Overview")

total_students = len(df)
average_marks = df["Final_Marks"].mean()
pass_count = (df["Final_Marks"] >= 40).sum()
fail_count = total_students - pass_count

pass_percentage = (pass_count / total_students) * 100

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Students", total_students)
c2.metric("Average Marks", f"{average_marks:.2f}")
c3.metric("Passing Percentage", f"{pass_percentage:.2f}%")
c4.metric("Failing Students", int(fail_count))

# ---------------- VISUALIZATION ----------------

st.header("3. Performance Visualization")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pass and Fail Count")
    st.bar_chart(df["Result"].value_counts())

with col2:
    st.subheader("Attendance vs Final Marks")
    st.scatter_chart(
        df,
        x="Attendance",
        y="Final_Marks"
    )

st.subheader("Marks Distribution")
st.hist_chart = None
st.bar_chart(
    df["Final_Marks"].value_counts().sort_index()
)

# ---------------- TRAIN MACHINE LEARNING MODELS ----------------

st.header("4. Machine Learning Models")

X = df[features]
y_marks = df["Final_Marks"]
y_result = (df["Final_Marks"] >= 40).astype(int)

if len(df) < 10 or y_result.nunique() < 2 or y_result.value_counts().min() < 2:
    st.warning(
        "Add more student records, including both pass and fail "
        "examples, to train and test the models."
    )
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y_marks,
    test_size=0.25,
    random_state=42
)

# Marks prediction model
marks_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

marks_model.fit(X_train, y_train)

marks_predictions = marks_model.predict(X_test)

mae = mean_absolute_error(y_test, marks_predictions)

# Pass/fail model
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_result,
    test_size=0.25,
    random_state=42,
    stratify=y_result
)

result_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

result_model.fit(X_train_c, y_train_c)

result_predictions = result_model.predict(X_test_c)

accuracy = accuracy_score(y_test_c, result_predictions)

col1, col2 = st.columns(2)

col1.metric("Marks Prediction MAE", f"{mae:.2f} marks")
col2.metric("Pass/Fail Model Accuracy", f"{accuracy * 100:.2f}%")

# ---------------- FEATURE IMPORTANCE ----------------

st.header("5. Important Performance Factors")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": marks_model.feature_importances_
}).sort_values("Importance", ascending=False)

st.bar_chart(
    importance.set_index("Feature")
)

st.write(importance)

# ---------------- USER INPUT ----------------

st.header("6. Predict Individual Student Performance")

st.write("Enter the student's details below.")

with st.form("student_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.number_input(
            "Attendance (%)", 0.0, 100.0, 75.0
        )

        assignment = st.number_input(
            "Assignment Score", 0.0, 100.0, 70.0
        )

        internal = st.number_input(
            "Internal Marks", 0.0, 100.0, 65.0
        )

    with col2:
        previous = st.number_input(
            "Previous Marks", 0.0, 100.0, 70.0
        )

        study_hours = st.number_input(
            "Study Hours Per Day", 0.0, 24.0, 3.0
        )

    submit = st.form_submit_button("Predict Performance")

if submit:

    student = pd.DataFrame([{
        "Attendance": attendance,
        "Assignment_Score": assignment,
        "Internal_Marks": internal,
        "Previous_Marks": previous,
        "Study_Hours": study_hours
    }])

    # Predict marks
    predicted_marks = marks_model.predict(student)[0]

    # Keep predicted marks in valid range
    predicted_marks = np.clip(predicted_marks, 0, 100)

    # Predict pass/fail
    predicted_result = result_model.predict(student)[0]

    # Probability of passing
    pass_probability = result_model.predict_proba(student)[0][
        list(result_model.classes_).index(1)
    ]

    st.subheader("Prediction Results")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Expected Final Marks",
        f"{predicted_marks:.2f}/100"
    )

    c2.metric(
        "Estimated Pass Probability",
        f"{pass_probability * 100:.2f}%"
    )

    c3.metric(
        "Predicted Result",
        "PASS" if predicted_result == 1 else "FAIL"
    )

    if predicted_result == 1:
        st.success("The model predicts that this student will PASS.")
    else:
        st.error("The model predicts that this student may FAIL.")

    st.caption(
        "These are estimates based on patterns in the uploaded dataset, "
        "not guaranteed exam results."
    )

# ---------------- END ----------------

st.header("7. Download Analyzed Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Cleaned Dataset",
    csv,
    "analyzed_students.csv",
    "text/csv"
)

st.success("Analysis completed!")
