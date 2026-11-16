
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Student Performance Insights",
    page_icon="📚",
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


# =====================================================
# 2. LIGHT ACADEMIC THEME
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #F8F7F2,
        #F0F4F1,
        #F1F0FA
    );
    color: #34443D;
}

[data-testid="stSidebar"] {
    background-color: #E9EFE9;
    border-right: 1px solid #DCE5DC;
}

h1, h2, h3 {
    color: #405B50 !important;
}

p, label {
    color: #52645B;
}

[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #E3EAE4;
    box-shadow: 0 4px 15px rgba(70, 90, 75, 0.06);
}

[data-testid="stForm"] {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #E3EAE4;
}

.stButton > button {
    background-color: #648B78;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 22px;
}

.stButton > button:hover {
    background-color: #4E735F;
    color: white;
}

[data-testid="stExpander"] {
    background: white;
    border-radius: 12px;
}

.student-card {
    background: white;
    padding: 14px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #E3EAE4;
    margin-bottom: 12px;
}

.student-icon {
    font-size: 35px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. TITLE
# =====================================================

st.title("📚 Student Performance Insights")

st.markdown(
    "### Understand your progress. Discover your potential."
)

st.caption(
    "An interactive data science platform to explore "
    "academic performance and predict student outcomes."
)


# =====================================================
# 4. LOAD TRAINING DATASET
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "student_performance.csv"
)

if not os.path.exists(DATA_PATH):
    st.error(
        "student_performance.csv not found. "
        "Place the training CSV in the same folder as app.py."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)

required_columns = [
    "Student_ID",
    "Attendance",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Marks",
    "Study_Hours",
    "Final_Marks"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns in training CSV: {missing_columns}")
    st.stop()


# =====================================================
# 5. DATA CLEANING
# =====================================================

df = df[required_columns].copy()

original_count = len(df)

df = df.drop_duplicates()

numeric_columns = FEATURES + ["Final_Marks"]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(subset=["Final_Marks"])

if df.empty:
    st.error("No valid student records found.")
    st.stop()

# Fill missing feature values with training medians
training_medians = {}

for col in FEATURES:

    median_value = df[col].median()

    if pd.isna(median_value):
        median_value = 0

    training_medians[col] = median_value

    df[col] = df[col].fillna(median_value)

# Clean marks and attendance
for col in [
    "Attendance",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Marks",
    "Final_Marks"
]:
    df[col] = df[col].clip(0, 100)

df["Study_Hours"] = df["Study_Hours"].clip(0, 24)

if len(df) < 2:
    st.error("At least two valid student records are required.")
    st.stop()

df["Result"] = np.where(
    df["Final_Marks"] >= PASS_MARK,
    "Pass",
    "Fail"
)

df["At_Risk"] = np.where(
    df["Final_Marks"] < PASS_MARK,
    "Yes",
    "No"
)


# =====================================================
# 6. MACHINE LEARNING MODELS
# =====================================================

X = df[FEATURES]

y_marks = df["Final_Marks"]

y_result = (
    df["Final_Marks"] >= PASS_MARK
).astype(int)

regressor = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)

classifier = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

mae = None
accuracy = None

class_counts = y_result.value_counts()

evaluation_available = (
    len(df) >= 20
    and len(class_counts) == 2
    and class_counts.min() >= 4
)

if evaluation_available:

    X_train, X_test, marks_train, marks_test, result_train, result_test = (
        train_test_split(
            X,
            y_marks,
            y_result,
            test_size=0.25,
            random_state=42,
            stratify=y_result
        )
    )

    # Evaluate regression
    evaluation_regressor = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    evaluation_regressor.fit(X_train, marks_train)

    marks_prediction = evaluation_regressor.predict(X_test)

    mae = mean_absolute_error(
        marks_test,
        marks_prediction
    )

    # Evaluate classification
    evaluation_classifier = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    evaluation_classifier.fit(X_train, result_train)

    result_prediction = evaluation_classifier.predict(X_test)

    accuracy = accuracy_score(
        result_test,
        result_prediction
    )

# Train final models using all available training records
regressor.fit(X, y_marks)
classifier.fit(X, y_result)


# =====================================================
# 7. HELPER FUNCTIONS
# =====================================================

def style_chart(fig, title):

    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font_color="#52645B",
        title_font_color="#405B50",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def predict_students(input_data):

    marks = regressor.predict(input_data[FEATURES])

    marks = np.clip(marks, 0, 100)

    probabilities = classifier.predict_proba(
        input_data[FEATURES]
    )

    classes = list(classifier.classes_)

    if 1 in classes:
        pass_index = classes.index(1)
        pass_probabilities = probabilities[:, pass_index] * 100
    else:
        pass_probabilities = np.zeros(len(input_data))

    predicted_classes = classifier.predict(
        input_data[FEATURES]
    )

    output = input_data.copy()

    output["Predicted_Marks"] = np.round(marks, 2)

    output["Pass_Probability"] = np.round(
        pass_probabilities,
        2
    )

    output["Predicted_Result"] = np.where(
        predicted_classes == 1,
        "Pass",
        "Fail"
    )

    return output


def show_importance_heatmap():

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": regressor.feature_importances_
    })

    importance = importance.set_index("Feature").T

    fig = px.imshow(
        importance,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="YlGn"
    )

    st.plotly_chart(
        style_chart(
            fig,
            "Prediction Basis: Random Forest Feature Importance"
        ),
        use_container_width=True
    )

    st.caption(
        "Higher values indicate features the Random Forest "
        "regression model relies on more when predicting marks. "
        "This is overall model importance, not a causal explanation "
        "or an individual student's exact contribution."
    )


# =====================================================
# 8. SIDEBAR
# =====================================================

st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "🏠 Dashboard",
        "📊 Performance Analysis",
        "🔮 Predict Performance",
        "📁 CSV Bulk Prediction",
        "💡 Academic Suggestions",
        "ℹ️ About Project"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Explore academic patterns, predict marks "
    "and discover ways to improve."
)


# =====================================================
# PAGE 1: DASHBOARD
# =====================================================

if page == "🏠 Dashboard":

    st.header("📈 Academic Overview")

    total_students = len(df)

    average_marks = df["Final_Marks"].mean()

    pass_percentage = (
        (df["Final_Marks"] >= PASS_MARK).mean() * 100
    )

    fail_count = int(
        (df["Final_Marks"] < PASS_MARK).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", total_students)
    c2.metric("Average Marks", f"{average_marks:.1f}")
    c3.metric("Pass Percentage", f"{pass_percentage:.1f}%")
    c4.metric("At-Risk Students", fail_count)

    st.divider()

    left, right = st.columns(2)

    with left:

        result_counts = (
            df["Result"]
            .value_counts()
            .rename_axis("Result")
            .reset_index(name="Students")
        )

        fig = px.pie(
            result_counts,
            names="Result",
            values="Students",
            hole=0.55,
            color="Result",
            color_discrete_map={
                "Pass": "#648B78",
                "Fail": "#E58B8B"
            }
        )

        st.plotly_chart(
            style_chart(fig, "Pass vs Fail Distribution"),
            use_container_width=True
        )

    with right:

        fig = px.histogram(
            df,
            x="Final_Marks",
            nbins=15,
            color_discrete_sequence=["#8E9FC8"]
        )

        st.plotly_chart(
            style_chart(fig, "Final Marks Distribution"),
            use_container_width=True
        )

    fig = px.scatter(
        df,
        x="Attendance",
        y="Final_Marks",
        color="Result",
        hover_data=["Study_Hours"]
    )

    st.plotly_chart(
        style_chart(fig, "Attendance vs Final Marks"),
        use_container_width=True
    )

    st.subheader("🤖 Model Evaluation")

    if evaluation_available:

        c1, c2 = st.columns(2)

        c1.metric("Mean Absolute Error", f"{mae:.2f}")

        c2.metric("Classification Accuracy", f"{accuracy * 100:.1f}%")

    else:

        st.warning(
            "Not enough records or class examples for reliable "
            "test evaluation. Models are trained on all available records."
        )

    st.subheader("🧹 Data Cleaning Summary")

    st.write(f"Original records: **{original_count}**")
    st.write(f"Valid records: **{len(df)}**")
    st.write(f"Records removed: **{original_count - len(df)}**")


# =====================================================
# PAGE 2: STATISTICS AND VISUALIZATIONS
# =====================================================

elif page == "📊 Performance Analysis":

    st.header("📊 Statistics & Visualizations")

    marks = df["Final_Marks"]

    average = marks.mean()
    median = marks.median()
    mode = marks.mode().iloc[0]

    highest = marks.max()
    lowest = marks.min()

    std_dev = marks.std()
    variance = marks.var()

    pass_percentage = (
        (marks >= PASS_MARK).mean() * 100
    )

    q1 = marks.quantile(0.25)
    q2 = marks.quantile(0.50)
    q3 = marks.quantile(0.75)

    st.subheader("📈 Statistical Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Mean", f"{average:.2f}")
    c2.metric("Median", f"{median:.2f}")
    c3.metric("Mode", f"{mode:.2f}")

    c4, c5, c6 = st.columns(3)

    c4.metric("Highest Marks", f"{highest:.2f}")
    c5.metric("Lowest Marks", f"{lowest:.2f}")
    c6.metric("Standard Deviation", f"{std_dev:.2f}")

    c7, c8, c9 = st.columns(3)

    c7.metric("Variance", f"{variance:.2f}")
    c8.metric("Pass Percentage", f"{pass_percentage:.1f}%")
    c9.metric("Total Students", len(df))

    with st.expander("View more statistical details"):

        statistics = pd.DataFrame({
            "Measure": [
                "Count",
                "Mean",
                "Median",
                "Mode",
                "Minimum",
                "Maximum",
                "Q1",
                "Q2",
                "Q3",
                "Standard Deviation",
                "Variance"
            ],
            "Value": [
                marks.count(),
                average,
                median,
                mode,
                lowest,
                highest,
                q1,
                q2,
                q3,
                std_dev,
                variance
            ]
        })

        st.dataframe(
            statistics,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader("1. Marks Distribution")

    fig = px.histogram(
        df,
        x="Final_Marks",
        nbins=15,
        color_discrete_sequence=["#648B78"]
    )

    st.plotly_chart(
        style_chart(fig, "Distribution of Student Marks"),
        use_container_width=True
    )

    st.subheader("2. Pass vs Fail")

    result_counts = (
        df["Result"]
        .value_counts()
        .rename_axis("Result")
        .reset_index(name="Students")
    )

    fig = px.pie(
        result_counts,
        names="Result",
        values="Students",
        hole=0.55,
        color="Result",
        color_discrete_map={
            "Pass": "#648B78",
            "Fail": "#E58B8B"
        }
    )

    st.plotly_chart(
        style_chart(fig, "Pass and Fail Distribution"),
        use_container_width=True
    )

    st.subheader("3. Attendance vs Final Marks")

    fig = px.scatter(
        df,
        x="Attendance",
        y="Final_Marks",
        color="Result",
        hover_data=["Study_Hours"]
    )

    st.plotly_chart(
        style_chart(fig, "Attendance vs Final Marks"),
        use_container_width=True
    )

    st.subheader("4. Correlation Heatmap")

    correlation = df[
        FEATURES + ["Final_Marks"]
    ].corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=-1,
        zmax=1
    )

    st.plotly_chart(
        style_chart(fig, "Correlation Between Academic Factors"),
        use_container_width=True
    )

    st.subheader("5. Average Academic Scores")

    factors = [
        "Attendance",
        "Assignment_Score",
        "Internal_Marks",
        "Previous_Marks"
    ]

    averages = pd.DataFrame({
        "Factor": factors,
        "Average": [df[col].mean() for col in factors]
    })

    fig = px.bar(
        averages,
        x="Factor",
        y="Average",
        color="Average",
        color_continuous_scale="Teal"
    )

    st.plotly_chart(
        style_chart(fig, "Average Academic Scores"),
        use_container_width=True
    )

    st.subheader("6. Study Hours Analysis")

    fig = px.box(
        df,
        y="Study_Hours",
        color="Result"
    )

    st.plotly_chart(
        style_chart(fig, "Study Hours by Result"),
        use_container_width=True
    )

    st.subheader("7. Prediction Basis Heatmap")

    show_importance_heatmap()


# =====================================================
# PAGE 3: INDIVIDUAL PREDICTION
# =====================================================

elif page == "🔮 Predict Performance":

    st.header("🔮 Predict Your Academic Performance")

    st.write(
        "Enter your academic details to estimate final marks "
        "and the probability of passing."
    )

    with st.form("prediction_form"):

        c1, c2 = st.columns(2)

        with c1:

            attendance = st.slider(
                "Attendance (%)",
                0, 100, 80
            )

            assignment = st.slider(
                "Assignment Score",
                0, 100, 70
            )

            internal = st.slider(
                "Internal Marks",
                0, 100, 65
            )

        with c2:

            previous = st.slider(
                "Previous Marks",
                0, 100, 60
            )

            hours = st.number_input(
                "Daily Study Hours",
                min_value=0.0,
                max_value=24.0,
                value=3.0,
                step=0.5
            )

        submitted = st.form_submit_button(
            "🚀 Predict My Performance"
        )

    if submitted:

        user_data = pd.DataFrame(
            [[attendance, assignment, internal, previous, hours]],
            columns=FEATURES
        )

        result = predict_students(user_data)

        predicted_marks = result["Predicted_Marks"].iloc[0]
        pass_probability = result["Pass_Probability"].iloc[0]
        predicted_result = result["Predicted_Result"].iloc[0]

        st.divider()

        st.subheader("Your Prediction Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Expected Final Marks",
            f"{predicted_marks:.1f}/100"
        )

        c2.metric(
            "Pass Probability",
            f"{pass_probability:.1f}%"
        )

        c3.metric(
            "Predicted Result",
            predicted_result
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pass_probability,
                title={"text": "Pass Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#648B78"},
                    "steps": [
                        {"range": [0, 40], "color": "#F6D7D7"},
                        {"range": [40, 70], "color": "#F8E9C5"},
                        {"range": [70, 100], "color": "#DCEDE1"}
                    ]
                }
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if predicted_result == "Pass":
            st.success("🧑‍🎓 Your predicted result is PASS.")
        else:
            st.warning("🧑‍🎓 Your predicted result is FAIL. Consider the improvement suggestions.")

        st.subheader("Why did the model make this prediction?")

        if st.checkbox("Show prediction basis heatmap"):

            show_importance_heatmap()

        st.info(
            "Predictions are estimates based on the training dataset, "
            "not guaranteed exam results."
        )


# =====================================================
# PAGE 4: CSV BULK PREDICTION
# =====================================================

elif page == "📁 CSV Bulk Prediction":

    st.header("📁 Upload CSV & Predict Multiple Students")

    st.write(
        "Upload a CSV file containing multiple student records. "
        "The application will predict marks and pass probability "
        "for every student."
    )

    st.info(
        "Your uploaded CSV does not need Final_Marks. "
        "The model predicts the marks for you."
    )

    st.subheader("Required CSV Columns")

    st.code(
        "Attendance, Assignment_Score, Internal_Marks, "
        "Previous_Marks, Study_Hours",
        language="text"
    )

    uploaded_file = st.file_uploader(
        "Choose a student CSV file",
        type=["csv"],
        help="Upload a CSV containing the five required academic input columns."
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(uploaded_file)

            st.success("CSV uploaded successfully!")

            missing = [
                col for col in FEATURES
                if col not in uploaded_df.columns
            ]

            if missing:

                st.error(
                    f"Your CSV is missing these columns: {missing}"
                )

            elif uploaded_df.empty:

                st.error("The uploaded CSV has no student records.")

            else:

                st.write(
                    f"Number of uploaded students: **{len(uploaded_df)}**"
                )

                # Keep Student_ID if available
                if "Student_ID" in uploaded_df.columns:
                    student_ids = uploaded_df["Student_ID"].astype(str)
                else:
                    student_ids = pd.Series(
                        [
                            f"Student {i + 1}"
                            for i in range(len(uploaded_df))
                        ]
                    )

                # Prepare model inputs
                input_data = uploaded_df[FEATURES].copy()

                for col in FEATURES:

                    input_data[col] = pd.to_numeric(
                        input_data[col],
                        errors="coerce"
                    )

                    input_data[col] = input_data[col].replace(
                        [np.inf, -np.inf],
                        np.nan
                    )

                    input_data[col] = input_data[col].fillna(
                        training_medians[col]
                    )

                # Apply same input limits as training data
                for col in [
                    "Attendance",
                    "Assignment_Score",
                    "Internal_Marks",
                    "Previous_Marks"
                ]:
                    input_data[col] = input_data[col].clip(0, 100)

                input_data["Study_Hours"] = (
                    input_data["Study_Hours"].clip(0, 24)
                )

                if st.button(
                    "🚀 Predict All Students",
                    type="primary"
                ):

                    predictions = predict_students(input_data)

                    predictions.insert(
                        0,
                        "Student_ID",
                        student_ids.values
                    )

                    st.session_state["bulk_predictions"] = predictions

        except Exception as e:

            st.error(f"Could not read the CSV file: {e}")

    # Display prediction results
    if "bulk_predictions" in st.session_state:

        predictions = st.session_state["bulk_predictions"]

        st.divider()

        st.subheader("📊 Bulk Prediction Results")

        total = len(predictions)

        passed = int(
            (predictions["Predicted_Result"] == "Pass").sum()
        )

        failed = total - passed

        average_predicted = predictions["Predicted_Marks"].mean()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Students", total)

        c2.metric("Predicted Pass", passed)

        c3.metric("Predicted Fail", failed)

        c4.metric(
            "Average Predicted Marks",
            f"{average_predicted:.1f}"
        )

        st.subheader("🧑‍🎓 Student Outcome Overview")

        st.write(
            "Each icon represents one student. "
            "Green indicates predicted pass; red indicates predicted fail."
        )

        # Display student icons in rows
        cards_per_row = 5

        for start in range(0, total, cards_per_row):

            columns = st.columns(cards_per_row)

            for j, col in enumerate(columns):

                index = start + j

                if index >= total:
                    break

                row = predictions.iloc[index]

                is_pass = row["Predicted_Result"] == "Pass"

                icon = "🟢" if is_pass else "🔴"

                result_text = "PASS" if is_pass else "FAIL"

                with col:

                    st.markdown(
                        f"""
                        <div class="student-card">
                            <div class="student-icon">🧑‍🎓</div>
                            <b>{row['Student_ID']}</b>
                            <h3>{icon} {result_text}</h3>
                            <p>Marks: {row['Predicted_Marks']}</p>
                            <p>Pass chance: {row['Pass_Probability']}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.divider()

        st.subheader("Detailed Prediction Table")

        st.dataframe(
            predictions,
            use_container_width=True,
            hide_index=True
        )

        # Download prediction results
        csv_data = predictions.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_data,
            file_name="student_prediction_results.csv",
            mime="text/csv"
        )

        st.divider()

        st.subheader("Prediction Explanation")

        if st.checkbox("Show model feature importance heatmap"):

            show_importance_heatmap()

        st.caption(
            "The heatmap shows overall model feature importance, "
            "not the exact reason for each individual student's result."
        )


# =====================================================
# PAGE 5: ACADEMIC SUGGESTIONS
# =====================================================

elif page == "💡 Academic Suggestions":

    st.header("💡 Your Personal Academic Improvement Plan")

    st.write(
        "Enter your academic details to receive suggestions "
        "for improving your performance."
    )

    with st.form("suggestion_form"):

        attendance = st.slider(
            "Your Attendance (%)",
            0, 100, 75
        )

        assignment = st.slider(
            "Assignment Score",
            0, 100, 60
        )

        internal = st.slider(
            "Internal Marks",
            0, 100, 55
        )

        previous = st.slider(
            "Previous Marks",
            0, 100, 55
        )

        hours = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=2.0,
            step=0.5
        )

        submitted = st.form_submit_button(
            "✨ Generate My Improvement Plan"
        )

    if submitted:

        user_data = pd.DataFrame(
            [[attendance, assignment, internal, previous, hours]],
            columns=FEATURES
        )

        result = predict_students(user_data)

        predicted = result["Predicted_Marks"].iloc[0]

        st.divider()

        st.subheader("Your Estimated Performance")

        st.metric(
            "Expected Final Marks",
            f"{predicted:.1f}/100"
        )

        st.subheader("Personalized Suggestions")

        suggestions = []

        if attendance < 75:
            suggestions.append(
                "📅 **Improve attendance:** Aim for at least "
                "75% attendance and avoid missing important classes."
            )

        if assignment < 50:
            suggestions.append(
                "📝 **Focus on assignments:** Complete assignments "
                "on time and clarify difficult concepts."
            )

        if internal < 50:
            suggestions.append(
                "📖 **Prepare for internal exams:** Revise class notes "
                "and practise previous questions regularly."
            )

        if previous < 50:
            suggestions.append(
                "🔁 **Strengthen your basics:** Identify weak subjects "
                "and spend extra time understanding fundamentals."
            )

        if hours < 2:
            suggestions.append(
                "⏰ **Increase focused study time:** Start with "
                "2 hours of distraction-free study daily."
            )

        elif hours >= 6:
            suggestions.append(
                "🌿 **Maintain a healthy routine:** Take breaks "
                "and balance study with sleep and exercise."
            )

        if not suggestions:
            suggestions.append(
                "🌟 **Good progress:** Maintain your routine, "
                "revise regularly and keep practising."
            )

        for suggestion in suggestions:
            st.markdown(suggestion)

        st.subheader("Recommended Daily Routine")

        routine = pd.DataFrame({
            "Activity": [
                "Revision",
                "Assignment Practice",
                "Problem Solving",
                "Break and Review"
            ],
            "Time": [
                "30 minutes",
                "45 minutes",
                "45 minutes",
                "15 minutes"
            ]
        })

        st.table(routine)


# =====================================================
# PAGE 6: ABOUT PROJECT
# =====================================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About Student Performance Insights")

    st.markdown("""
    ### 🎯 Project Objective

    This project uses data science and machine learning to
    understand student academic performance and identify
    students who may need additional academic support.

    ### 🔍 Main Features

    - Data cleaning and preprocessing
    - Statistical analysis
    - Interactive dashboard
    - Academic performance visualizations
    - Individual student prediction
    - CSV bulk student prediction
    - Student pass/fail icon visualization
    - Prediction explanation heatmap
    - Personalized academic suggestions

    ### 📊 Statistical Methods

    - Mean
    - Median
    - Mode
    - Standard deviation
    - Variance
    - Quartiles
    - Correlation

    ### 🤖 Machine Learning Algorithms

    **1. Random Forest Regressor**

    Predicts expected final marks.

    **2. Random Forest Classifier**

    Predicts whether a student is likely to pass or fail.

    ### 🛠️ Technologies Used

    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Plotly
    - Scikit-learn

    ### 📌 Dataset Information

    The application uses anonymized or synthetic student
    performance data. Student IDs are not used as model inputs.

    ### ⚠️ Important Note

    Predictions are estimates, not guaranteed outcomes.
    Model performance depends on dataset quality, size
    and representativeness.
    """)

    st.success(
        "Built to turn academic data into useful insights."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <center>
        📚 Student Performance Insights |
        Data Science Project
    </center>
    """,
    unsafe_allow_html=True
)

