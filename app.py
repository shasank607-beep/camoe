
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
# 4. LOAD DATASET
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "student_performance.csv"
)

if not os.path.exists(DATA_PATH):
    st.error(
        "student_performance.csv not found. "
        "Place the CSV file in the same folder as app.py."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)


# =====================================================
# 5. DATA CLEANING
# =====================================================

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
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

df = df[required_columns].copy()

original_count = len(df)

# Remove duplicate records
df = df.drop_duplicates()

# Convert numeric columns
numeric_columns = [
    "Attendance",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Marks",
    "Study_Hours",
    "Final_Marks"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove rows without final marks
df = df.dropna(subset=["Final_Marks"])

if df.empty:
    st.error("No valid student records found.")
    st.stop()

# Fill missing predictor values with median
for col in numeric_columns:
    if col != "Final_Marks":

        median_value = df[col].median()

        if pd.isna(median_value):
            median_value = 0

        df[col] = df[col].fillna(median_value)

# Keep marks and attendance between 0 and 100
mark_columns = [
    "Attendance",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Marks",
    "Final_Marks"
]

for col in mark_columns:
    df[col] = df[col].clip(0, 100)

df["Study_Hours"] = df["Study_Hours"].clip(0, 24)

if len(df) < 2:
    st.error("At least two valid records are needed.")
    st.stop()

# Derive pass/fail
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
# 6. MACHINE LEARNING
# =====================================================

features = [
    "Attendance",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Marks",
    "Study_Hours"
]

X = df[features]

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

# Evaluate only when there are enough examples
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

    # Evaluate regression model
    evaluation_regressor = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    evaluation_regressor.fit(
        X_train,
        marks_train
    )

    marks_prediction = evaluation_regressor.predict(X_test)

    mae = mean_absolute_error(
        marks_test,
        marks_prediction
    )

    # Evaluate classification model
    evaluation_classifier = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    evaluation_classifier.fit(
        X_train,
        result_train
    )

    result_prediction = evaluation_classifier.predict(X_test)

    accuracy = accuracy_score(
        result_test,
        result_prediction
    )

# Train final models using all available data
regressor.fit(X, y_marks)

classifier.fit(X, y_result)


# =====================================================
# 7. COMMON CHART DESIGN
# =====================================================

def style_chart(fig, title):

    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font_color="#52645B",
        title_font_color="#405B50",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig


# =====================================================
# 8. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "🏠 Dashboard",
        "📊 Performance Analysis",
        "🔮 Predict Performance",
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

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Students",
        total_students
    )

    c2.metric(
        "Average Marks",
        f"{average_marks:.1f}"
    )

    c3.metric(
        "Pass Percentage",
        f"{pass_percentage:.1f}%"
    )

    c4.metric(
        "At-Risk Students",
        fail_count
    )

    st.divider()

    left, right = st.columns(2)

    # Pass fail chart
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

    # Marks histogram
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

    # Attendance scatter plot
    fig = px.scatter(
        df,
        x="Attendance",
        y="Final_Marks",
        color="Result",
        hover_data=["Study_Hours"],
        color_discrete_map={
            "Pass": "#648B78",
            "Fail": "#E58B8B"
        }
    )

    st.plotly_chart(
        style_chart(fig, "Attendance vs Final Marks"),
        use_container_width=True
    )

    # Model evaluation
    st.subheader("🤖 Machine Learning Evaluation")

    if evaluation_available:

        c1, c2 = st.columns(2)

        c1.metric(
            "Mean Absolute Error",
            f"{mae:.2f}"
        )

        c2.metric(
            "Classification Accuracy",
            f"{accuracy * 100:.1f}%"
        )

        st.caption(
            "Evaluation uses a separate test set. "
            "The final prediction models are then trained "
            "on the complete dataset."
        )

    else:

        st.warning(
            "The dataset is too small or does not have enough "
            "pass and fail examples for reliable test evaluation. "
            "Models are trained using all available records."
        )

    # Cleaning information
    st.subheader("🧹 Data Cleaning Summary")

    removed_records = original_count - len(df)

    st.write(
        f"Original records: **{original_count}**"
    )

    st.write(
        f"Valid records after cleaning: **{len(df)}**"
    )

    st.write(
        f"Records removed: **{removed_records}**"
    )


# =====================================================
# PAGE 2: STATISTICS AND VISUALIZATIONS
# =====================================================

elif page == "📊 Performance Analysis":

    st.header("📊 Statistics & Visualizations")

    st.write(
        "Explore student performance using statistical "
        "measures and interactive charts."
    )

    # -----------------------------------------------
    # STATISTICAL CALCULATIONS
    # -----------------------------------------------

    marks = df["Final_Marks"]

    average = marks.mean()
    median = marks.median()
    mode = marks.mode().iloc[0]

    highest = marks.max()
    lowest = marks.min()

    standard_deviation = marks.std()
    variance = marks.var()

    q1 = marks.quantile(0.25)
    q2 = marks.quantile(0.50)
    q3 = marks.quantile(0.75)

    pass_percentage = (
        (marks >= PASS_MARK).mean() * 100
    )

    # -----------------------------------------------
    # STATISTICAL METRIC CARDS
    # -----------------------------------------------

    st.subheader("📈 Statistical Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Mean / Average",
        f"{average:.2f}"
    )

    c2.metric(
        "Median",
        f"{median:.2f}"
    )

    c3.metric(
        "Mode",
        f"{mode:.2f}"
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "Highest Marks",
        f"{highest:.2f}"
    )

    c5.metric(
        "Lowest Marks",
        f"{lowest:.2f}"
    )

    c6.metric(
        "Standard Deviation",
        f"{standard_deviation:.2f}"
    )

    c7, c8, c9 = st.columns(3)

    c7.metric(
        "Variance",
        f"{variance:.2f}"
    )

    c8.metric(
        "Pass Percentage",
        f"{pass_percentage:.1f}%"
    )

    c9.metric(
        "Total Students",
        len(df)
    )

    # More statistical information
    with st.expander("View more statistical details"):

        statistics = pd.DataFrame({
            "Measure": [
                "Count",
                "Mean",
                "Median",
                "Mode",
                "Minimum",
                "Maximum",
                "25th Percentile (Q1)",
                "50th Percentile (Q2)",
                "75th Percentile (Q3)",
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
                standard_deviation,
                variance
            ]
        })

        st.dataframe(
            statistics,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # -----------------------------------------------
    # CHART 1: HISTOGRAM
    # -----------------------------------------------

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

    st.caption(
        "A histogram shows how many students fall into "
        "different marks ranges."
    )

    # -----------------------------------------------
    # CHART 2: PASS FAIL PIE CHART
    # -----------------------------------------------

    st.subheader("2. Pass vs Fail Distribution")

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
        style_chart(fig, "Pass and Fail Percentage"),
        use_container_width=True
    )

    # -----------------------------------------------
    # CHART 3: SCATTER PLOT
    # -----------------------------------------------

    st.subheader("3. Attendance vs Final Marks")

    fig = px.scatter(
        df,
        x="Attendance",
        y="Final_Marks",
        color="Result",
        hover_data=["Study_Hours"],
        color_discrete_map={
            "Pass": "#648B78",
            "Fail": "#E58B8B"
        }
    )

    st.plotly_chart(
        style_chart(fig, "Attendance vs Final Marks"),
        use_container_width=True
    )

    st.caption(
        "This chart helps explore whether attendance "
        "is associated with final marks."
    )

    # -----------------------------------------------
    # CHART 4: CORRELATION HEATMAP
    # -----------------------------------------------

    st.subheader("4. Correlation Heatmap")

    correlation_columns = [
        "Attendance",
        "Assignment_Score",
        "Internal_Marks",
        "Previous_Marks",
        "Study_Hours",
        "Final_Marks"
    ]

    correlation = df[correlation_columns].corr()

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

    st.caption(
        "Correlation values range from -1 to +1. "
        "Correlation does not prove causation."
    )

    # -----------------------------------------------
    # CHART 5: AVERAGE SCORES
    # -----------------------------------------------

    st.subheader("5. Academic Factor Comparison")

    factors = [
        "Attendance",
        "Assignment_Score",
        "Internal_Marks",
        "Previous_Marks"
    ]

    averages = pd.DataFrame({
        "Factor": factors,
        "Average": [
            df[col].mean()
            for col in factors
        ]
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

    # -----------------------------------------------
    # CHART 6: STUDY HOURS BOXPLOT
    # -----------------------------------------------

    st.subheader("6. Study Hours Analysis")

    fig = px.box(
        df,
        y="Study_Hours",
        color="Result",
        color_discrete_map={
            "Pass": "#648B78",
            "Fail": "#E58B8B"
        }
    )

    st.plotly_chart(
        style_chart(fig, "Study Hours Distribution by Result"),
        use_container_width=True
    )

    # -----------------------------------------------
    # CHART 7: FEATURE IMPORTANCE
    # -----------------------------------------------

    st.subheader("7. Machine Learning Feature Importance")

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": regressor.feature_importances_
    }).sort_values(
        "Importance",
        ascending=True
    )

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Teal"
    )

    st.plotly_chart(
        style_chart(fig, "Factors Used by the Prediction Model"),
        use_container_width=True
    )

    st.caption(
        "Feature importance explains how much each variable "
        "contributes to the Random Forest model. It does not "
        "prove that the variable causes better marks."
    )


# =====================================================
# PAGE 3: PREDICT PERFORMANCE
# =====================================================

elif page == "🔮 Predict Performance":

    st.header("🔮 Predict Your Academic Performance")

    st.write(
        "Enter your academic details to estimate "
        "your final marks and probability of passing."
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
            [[
                attendance,
                assignment,
                internal,
                previous,
                hours
            ]],
            columns=features
        )

        predicted_marks = float(
            np.clip(
                regressor.predict(user_data)[0],
                0,
                100
            )
        )

        prediction = classifier.predict(user_data)[0]

        probabilities = classifier.predict_proba(user_data)[0]

        class_positions = list(classifier.classes_)

        if 1 in class_positions:

            pass_probability = float(
                probabilities[class_positions.index(1)] * 100
            )

        else:

            pass_probability = 0.0

        st.divider()

        st.subheader("Your Prediction Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Expected Final Marks",
            f"{predicted_marks:.1f}/100"
        )

        c2.metric(
            "Probability of Passing",
            f"{pass_probability:.1f}%"
        )

        c3.metric(
            "Predicted Result",
            "PASS" if prediction == 1 else "FAIL"
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
                        {
                            "range": [0, 40],
                            "color": "#F6D7D7"
                        },
                        {
                            "range": [40, 70],
                            "color": "#F8E9C5"
                        },
                        {
                            "range": [70, 100],
                            "color": "#DCEDE1"
                        }
                    ]
                }
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "These predictions are estimates based on the "
            "available dataset. They are not guaranteed results."
        )


# =====================================================
# PAGE 4: ACADEMIC SUGGESTIONS
# =====================================================

elif page == "💡 Academic Suggestions":

    st.header("💡 Your Personal Academic Improvement Plan")

    st.write(
        "Enter your current academic details to receive "
        "suggestions for improving your performance."
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
            [[
                attendance,
                assignment,
                internal,
                previous,
                hours
            ]],
            columns=features
        )

        predicted = float(
            np.clip(
                regressor.predict(user_data)[0],
                0,
                100
            )
        )

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
                "📖 **Prepare for internal exams:** Revise class "
                "notes and practise previous questions regularly."
            )

        if previous < 50:

            suggestions.append(
                "🔁 **Strengthen your basics:** Identify weak "
                "subjects and spend extra time understanding fundamentals."
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
# PAGE 5: ABOUT PROJECT
# =====================================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About Student Performance Insights")

    st.markdown("""
    ### 🎯 Project Objective

    This project uses data science and machine learning
    to understand student academic performance and identify
    students who may need additional academic support.

    ### 🔍 Main Features

    - Data cleaning and preprocessing
    - Statistical analysis
    - Interactive academic dashboard
    - Student performance visualizations
    - Final marks prediction
    - Pass/fail probability estimation
    - Personalized academic suggestions
    - Machine learning feature importance

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

    Predicts expected final marks using academic information.

    **2. Random Forest Classifier**

    Estimates whether a student is likely to pass or fail.

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

    Predictions are educational estimates, not guaranteed
    outcomes. Model performance depends on dataset quality,
    size and representativeness.
    """)

    st.success(
        "Built to turn academic data into useful insights."
    )


# =====================================================
# 9. FOOTER
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
