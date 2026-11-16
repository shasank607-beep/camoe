
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Insights",
    page_icon="📚",
    layout="wide"
)

PASS_MARK = 40

# --------------------------------------------------
# LIGHT ACADEMIC THEME
# --------------------------------------------------

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

p, label, span {
    color: #52645B;
}

[data-testid="stMetric"] {
    background: white;
    padding: 20px;
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

div[data-baseweb="input"] input {
    background-color: white;
}

[data-testid="stExpander"] {
    background: white;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📚 Student Performance Insights")
st.markdown(
    "### Understand your progress. Discover your potential."
)

st.caption(
    "An interactive data science platform for exploring "
    "academic performance and predicting student outcomes."
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "student_performance.csv")

if not os.path.exists(DATA_PATH):
    st.error(
        "student_performance.csv not found. "
        "Place the CSV file in the same folder as app.py."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

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
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

df = df[required_columns].copy()

df = df.drop_duplicates()

numeric_columns = [
    "Attendance",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Marks",
    "Study_Hours",
    "Final_Marks"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf], np.nan
)

# Remove records without a target value
df = df.dropna(subset=["Final_Marks"])

# Fill missing input values using median
for col in numeric_columns:
    if col != "Final_Marks":
        df[col] = df[col].fillna(df[col].median())

# Clean unrealistic values
for col in [
    "Attendance",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Marks",
    "Final_Marks"
]:
    df[col] = df[col].clip(0, 100)

df["Study_Hours"] = df["Study_Hours"].clip(lower=0)

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

# --------------------------------------------------
# MACHINE LEARNING
# --------------------------------------------------

features = [
    "Attendance",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Marks",
    "Study_Hours"
]

X = df[features]
y_marks = df["Final_Marks"]
y_result = (df["Final_Marks"] >= PASS_MARK).astype(int)

regressor = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)

classifier = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

evaluation_available = (
    len(df) >= 12
    and y_result.value_counts().min() >= 3
)

mae = None
accuracy = None

if evaluation_available:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_marks,
        test_size=0.25,
        random_state=42
    )

    regressor.fit(X_train, y_train)

    marks_predictions = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, marks_predictions)

    X_train_c, X_test_c, yc_train, yc_test = train_test_split(
        X,
        y_result,
        test_size=0.25,
        random_state=42,
        stratify=y_result
    )

    classifier.fit(X_train_c, yc_train)

    result_predictions = classifier.predict(X_test_c)
    accuracy = accuracy_score(yc_test, result_predictions)

else:
    regressor.fit(X, y_marks)
    classifier.fit(X, y_result)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

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

# --------------------------------------------------
# COMMON CHART STYLE
# --------------------------------------------------

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


# ==================================================
# PAGE 1: DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    st.header("📈 Academic Overview")

    total_students = len(df)
    average_marks = df["Final_Marks"].mean()

    pass_percentage = (
        (df["Final_Marks"] >= PASS_MARK).mean() * 100
    )

    fail_count = int((df["Final_Marks"] < PASS_MARK).sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", total_students)
    col2.metric("Average Marks", f"{average_marks:.1f}")
    col3.metric("Pass Percentage", f"{pass_percentage:.1f}%")
    col4.metric("At-Risk Students", fail_count)

    st.divider()

    left, right = st.columns(2)

    with left:

        result_counts = df["Result"].value_counts().reset_index()
        result_counts.columns = ["Result", "Students"]

        fig = px.pie(
            result_counts,
            names="Result",
            values="Students",
            hole=0.55,
            title="Pass vs Fail Distribution"
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
            title="Distribution of Final Marks"
        )

        st.plotly_chart(
            style_chart(fig, "Distribution of Final Marks"),
            use_container_width=True
        )

    fig = px.scatter(
        df,
        x="Attendance",
        y="Final_Marks",
        color="Result",
        hover_data=["Study_Hours"],
        title="Attendance vs Final Marks"
    )

    st.plotly_chart(
        style_chart(fig, "Attendance vs Final Marks"),
        use_container_width=True
    )

    st.subheader("Model Evaluation")

    if evaluation_available:

        c1, c2 = st.columns(2)

        c1.metric("Mean Absolute Error", f"{mae:.2f}")
        c2.metric("Model Accuracy", f"{accuracy * 100:.1f}%")

        st.caption(
            "Metrics are calculated on a held-out test set. "
            "Small or synthetic datasets can give unstable results."
        )

    else:

        st.warning(
            "The dataset is too small or has too few examples "
            "in one class for reliable test evaluation. "
            "Models are trained on all available records."
        )


# ==================================================
# PAGE 2: PERFORMANCE ANALYSIS
# ==================================================

elif page == "📊 Performance Analysis":

    st.header("📊 Explore Academic Performance")

    selected_factor = st.selectbox(
        "Choose an academic factor",
        features
    )

    fig = px.scatter(
        df,
        x=selected_factor,
        y="Final_Marks",
        trendline="ols",
        title=f"{selected_factor} vs Final Marks"
    )

    st.plotly_chart(
        style_chart(fig, f"{selected_factor} vs Final Marks"),
        use_container_width=True
    )

    st.divider()

    st.subheader("Average Marks by Factor")

    factor_averages = pd.DataFrame({
        "Factor": features,
        "Average": [
            df[col].mean() for col in features
        ]
    })

    fig = px.bar(
        factor_averages,
        x="Factor",
        y="Average",
        color="Average",
        title="Average Value of Academic Factors"
    )

    st.plotly_chart(
        style_chart(fig, "Academic Factor Averages"),
        use_container_width=True
    )

    st.subheader("Correlation Heatmap")

    corr = df[features + ["Final_Marks"]].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Correlation Between Academic Factors"
    )

    st.plotly_chart(
        style_chart(fig, "Academic Correlation Heatmap"),
        use_container_width=True
    )

    st.subheader("Study Hours Analysis")

    fig = px.box(
        df,
        y="Study_Hours",
        color="Result",
        title="Study Hours Distribution"
    )

    st.plotly_chart(
        style_chart(fig, "Study Hours Distribution"),
        use_container_width=True
    )

    st.subheader("What influences predictions?")

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": regressor.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )

    st.plotly_chart(
        style_chart(fig, "Feature Importance"),
        use_container_width=True
    )

    st.caption(
        "Feature importance shows how much a feature contributes "
        "to the model's predictions. It does not prove causation."
    )


# ==================================================
# PAGE 3: PREDICT PERFORMANCE
# ==================================================

elif page == "🔮 Predict Performance":

    st.header("🔮 Predict Your Academic Performance")

    st.write(
        "Enter your academic details to estimate your final marks "
        "and probability of passing."
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
            columns=features
        )

        predicted_marks = float(
            regressor.predict(user_data)[0]
        )

        predicted_marks = float(
            np.clip(predicted_marks, 0, 100)
        )

        probabilities = classifier.predict_proba(user_data)[0]

        class_positions = list(classifier.classes_)
        pass_probability = float(
            probabilities[class_positions.index(1)] * 100
        )

        prediction = classifier.predict(user_data)[0]

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

        fig = go.Figure(go.Indicator(
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
        ))

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "These are estimates based on patterns in the available "
            "dataset, not guaranteed exam results."
        )


# ==================================================
# PAGE 4: ACADEMIC SUGGESTIONS
# ==================================================

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
            [[attendance, assignment, internal, previous, hours]],
            columns=features
        )

        predicted = float(
            np.clip(regressor.predict(user_data)[0], 0, 100)
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
                "📅 **Improve attendance:** Aim for at least 75% "
                "attendance and avoid missing important classes."
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


# ==================================================
# PAGE 5: ABOUT PROJECT
# ==================================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About Student Performance Insights")

    st.markdown("""
    ### 🎯 Project Objective

    This project uses data science and machine learning to
    understand student academic performance and identify
    students who may need additional academic support.

    ### 🔍 Main Features

    - Data cleaning and preprocessing
    - Academic performance dashboard
    - Interactive data visualizations
    - Student marks prediction
    - Pass/fail probability estimation
    - Academic improvement suggestions
    - Machine learning feature importance

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

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    "<center>📚 Student Performance Insights | "
    "Data Science Project</center>",
    unsafe_allow_html=True
)
