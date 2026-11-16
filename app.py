


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import hashlib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, mean_absolute_error
)
from twilio.rest import Client
import streamlit as st

def send_sms(phone, message):

    client = Client(
        st.secrets["TWILIO_ACCOUNT_SID"],
        st.secrets["TWILIO_AUTH_TOKEN"]
    )

    result = client.messages.create(
        body=message,
        from_=st.secrets["TWILIO_PHONE_NUMBER"],
        to=phone
    )

    return result.sid

# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="              Student Performance & Placement Insights",
    page_icon="🎓",
    layout="wide"
)

PASS_MARK = 40

# Placement probability tiers (in %)
READY_THRESHOLD = 75
CUSP_THRESHOLD = 50

PAGE_SIZE = 30

FEATURES = [
    "Attendance",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Marks",
    "Study_Hours"
]

REQUIRED_COLUMNS = FEATURES + ["Final_Marks"]

BRANCHES = ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"]

# Extra inputs used by the placement model
PLACEMENT_NUMERIC = [
    "CGPA",
    "Backlogs",
    "Tenth_Pct",
    "Twelfth_Pct",
    "Coding_Score",
    "Programming_Languages",
    "Certifications",
    "Projects",
    "Internships",
    "Hackathons",
    "Communication_Score",
    "Leadership_Roles",
    "Extracurricular_Activities",
    "Placement_Training"
]

COUNT_COLUMNS = [
    "Backlogs", "Programming_Languages", "Certifications", "Projects",
    "Internships", "Hackathons", "Leadership_Roles",
    "Extracurricular_Activities"
]

PLACEMENT_LIMITS = {
    "CGPA": (0, 10),
    "Backlogs": (0, 15),
    "Tenth_Pct": (0, 100),
    "Twelfth_Pct": (0, 100),
    "Coding_Score": (0, 100),
    "Programming_Languages": (0, 15),
    "Certifications": (0, 20),
    "Projects": (0, 20),
    "Internships": (0, 10),
    "Hackathons": (0, 15),
    "Communication_Score": (0, 100),
    "Leadership_Roles": (0, 10),
    "Extracurricular_Activities": (0, 15),
    "Placement_Training": (0, 1)
}

REQUIRED_PLACEMENT_COLUMNS = PLACEMENT_NUMERIC + ["Branch", "Placed"]

PLACEMENT_COLUMNS = (
    FEATURES
    + PLACEMENT_NUMERIC
    + [f"Branch_{b}" for b in BRANCHES]
)

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "placement_dataset.csv"
)

STATUS_COLORS = {
    "Placement Ready": "#10B981",
    "Almost There": "#F97316",
    "Needs Preparation": "#F43F5E"
}

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
.placement-banner {
    border-radius: 24px;
    padding: 30px 20px;
    text-align: center;
    color: white;
    animation: glow 2.6s ease-in-out infinite;
}
@keyframes glow {
    0%, 100% { box-shadow: 0 8px 22px rgba(0,0,0,0.15); }
    50% { box-shadow: 0 14px 38px rgba(0,0,0,0.32); }
}
.pb-emoji { font-size: 66px; line-height: 1.1; }
.pb-title {
    font-size: 40px;
    font-weight: 800;
    letter-spacing: 3px;
    margin: 6px 0;
}
.pb-sub { font-size: 16px; opacity: 0.96; }
.pb-prob {
    display: inline-block;
    margin-top: 14px;
    padding: 7px 20px;
    background: rgba(255,255,255,0.24);
    border-radius: 999px;
    font-weight: 700;
    font-size: 18px;
}
.chip {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13px;
    margin: 4px;
    color: white;
}
.pill {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    color: white;
    font-weight: 700;
    font-size: 13px;
    margin: 4px 0 8px 0;
}
.criteria-card {
    background: white;
    border-radius: 14px;
    padding: 12px 8px;
    border: 2px solid;
    text-align: center;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


def render_html(markup):
    """Render HTML safely (strips indentation and newlines so that
    Markdown never treats indented tags as a code block)."""
    flat = "".join(line.strip() for line in markup.splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# =====================================================
# SAMPLE INDIAN STUDENT PROFILES
# =====================================================

MALE_NAMES = [
    "Aarav Sharma", "Arjun Reddy", "Rohan Kumar",
    "Aditya Shetty", "Vikram Rao", "Karthik Gowda",
    "Rahul Verma", "Sanjay Patil", "Ishaan Mehta"
]

FEMALE_NAMES = [
    "Ananya Sharma", "Priya Reddy", "Diya Kumar",
    "Sneha Shetty", "Aishwarya Rao", "Kavya Gowda",
    "Meera Verma", "Pooja Patil", "Isha Mehta"
]


def generate_profile(identifier):
    value = int(
        hashlib.md5(str(identifier).encode()).hexdigest(), 16
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
# PLACEMENT HELPERS
# =====================================================

def build_placement_matrix(df):
    """Numeric inputs + one-hot Branch, in a fixed column order."""
    X = df[FEATURES + PLACEMENT_NUMERIC].apply(
        pd.to_numeric, errors="coerce"
    ).astype(float)

    branch_values = df["Branch"].astype(str).values

    for branch in BRANCHES:
        X[f"Branch_{branch}"] = (
            branch_values == branch
        ).astype(float)

    return X[PLACEMENT_COLUMNS]


def positive_probability(model, X):
    probabilities = model.predict_proba(X)
    positions = np.where(model.classes_ == 1)[0]

    if len(positions) == 0:
        return np.zeros(len(X))

    return probabilities[:, positions[0]]


def placement_tier(probability):
    if probability >= READY_THRESHOLD:
        return {
            "short": "Placement Ready",
            "label": "PLACEMENT READY",
            "emoji": "🏆",
            "c1": "#047857",
            "c2": "#10B981",
            "message": "Strong academics, skills and experience. "
                       "Start applying to the top drives!"
        }

    if probability >= CUSP_THRESHOLD:
        return {
            "short": "Almost There",
            "label": "ALMOST THERE",
            "emoji": "🚀",
            "c1": "#D97706",
            "c2": "#F97316",
            "message": "Good chance, but a few targeted improvements "
                       "can tip the odds in your favour."
        }

    return {
        "short": "Needs Preparation",
        "label": "NEEDS PREPARATION",
        "emoji": "🚨",
        "c1": "#DC2626",
        "c2": "#F43F5E",
        "message": "Focus on the weak areas below before the next "
                   "placement season."
    }


def get_placement_criteria(student):
    """Fundamental placement criteria.
    CORE = common recruiter eligibility cut-offs.
    SKILL = employability benchmarks."""

    has_experience = (
        student["Internships"] >= 1 or student["Projects"] >= 2
    )

    return [
        ("CGPA ≥ 6.0", student["CGPA"] >= 6.0,
         f"{student['CGPA']:.2f}", True),
        ("No active backlogs", student["Backlogs"] == 0,
         f"{int(student['Backlogs'])}", True),
        ("10th ≥ 60%", student["Tenth_Pct"] >= 60,
         f"{student['Tenth_Pct']:.1f}%", True),
        ("12th ≥ 60%", student["Twelfth_Pct"] >= 60,
         f"{student['Twelfth_Pct']:.1f}%", True),
        ("Attendance ≥ 75%", student["Attendance"] >= 75,
         f"{student['Attendance']:.0f}%", False),
        ("Coding score ≥ 50", student["Coding_Score"] >= 50,
         f"{student['Coding_Score']:.0f}", False),
        ("Communication ≥ 50", student["Communication_Score"] >= 50,
         f"{student['Communication_Score']:.0f}", False),
        ("Internship or 2+ projects", has_experience,
         f"{int(student['Internships'])} intern · "
         f"{int(student['Projects'])} proj", False)
    ]


def readiness_scores(row):
    """Eight employability axes on a 0-100 scale."""
    return [
        min(row["CGPA"] * 10, 100),
        row["Coding_Score"],
        row["Communication_Score"],
        min(row["Projects"] / 5 * 100, 100),
        min(row["Internships"] / 3 * 100, 100),
        min(row["Certifications"] / 4 * 100, 100),
        min(row["Hackathons"] / 3 * 100, 100),
        min(row["Programming_Languages"] / 5 * 100, 100)
    ]


READINESS_LABELS = [
    "CGPA", "Coding", "Communication", "Projects",
    "Internships", "Certifications", "Hackathons", "Languages"
]


# =====================================================
# LOAD TRAINING DATA
# =====================================================

@st.cache_data
def load_training_data():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "placement_dataset.csv not found beside app.py"
        )

    data = pd.read_csv(DATASET_PATH)

    needed = REQUIRED_COLUMNS + REQUIRED_PLACEMENT_COLUMNS

    missing = [
        col for col in needed
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    has_package = "Package_LPA" in data.columns

    keep = needed + (["Package_LPA"] if has_package else [])
    data = data[keep].copy()

    numeric_cols = (
        REQUIRED_COLUMNS + PLACEMENT_NUMERIC + ["Placed"]
        + (["Package_LPA"] if has_package else [])
    )

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Final_Marks", "Placed"])

    if len(data) < 2:
        raise ValueError("At least two valid training records are required.")

    for col in FEATURES + PLACEMENT_NUMERIC:
        median = data[col].median()
        data[col] = data[col].fillna(
            0 if pd.isna(median) else median
        )

    for col in [
        "Attendance", "Assignment_Score",
        "Internal_Marks", "Previous_Marks", "Final_Marks"
    ]:
        data[col] = data[col].clip(0, 100)

    data["Study_Hours"] = data["Study_Hours"].clip(0, 24)

    for col, (low, high) in PLACEMENT_LIMITS.items():
        data[col] = data[col].clip(low, high)

    data["Branch"] = data["Branch"].astype(str).str.strip()
    valid_branch = data["Branch"].isin(BRANCHES)
    fallback = (
        data.loc[valid_branch, "Branch"].mode().iloc[0]
        if valid_branch.any() else BRANCHES[0]
    )
    data.loc[~valid_branch, "Branch"] = fallback

    data["Placed"] = (data["Placed"] >= 1).astype(int)

    if data["Placed"].nunique() < 2:
        raise ValueError(
            "Placed column must contain both placed (1) and "
            "not placed (0) students."
        )

    data["Placed_Label"] = np.where(
        data["Placed"] == 1, "Placed", "Not Placed"
    )

    return data


# =====================================================
# TRAIN MACHINE LEARNING MODELS
# =====================================================

PLACEMENT_RF_PARAMS = dict(
    n_estimators=300,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


@st.cache_resource
def train_models(data):

    # ----- Academic models (unchanged) -----

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

    # ----- Placement classifier -----

    Xp = build_placement_matrix(data)
    yp = data["Placed"]

    metrics = {}

    if len(data) >= 50 and yp.value_counts().min() >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            Xp, yp, test_size=0.2, stratify=yp, random_state=42
        )

        probe = RandomForestClassifier(**PLACEMENT_RF_PARAMS)
        probe.fit(X_train, y_train)

        probability = positive_probability(probe, X_test)
        predicted = (probability >= 0.5).astype(int)

        metrics = {
            "Accuracy": accuracy_score(y_test, predicted),
            "F1 score": f1_score(y_test, predicted),
            "ROC-AUC": roc_auc_score(y_test, probability),
            "Test rows": len(y_test)
        }

    placement_model = RandomForestClassifier(**PLACEMENT_RF_PARAMS)
    placement_model.fit(Xp, yp)

    # ----- Package regressor (placed students only) -----

    package_model = None
    package_mae = None

    if "Package_LPA" in data.columns:
        placed = data[
            (data["Placed"] == 1) & data["Package_LPA"].notna()
        ]

        if len(placed) >= 30:
            Xk = build_placement_matrix(placed)
            yk = placed["Package_LPA"]

            Xk_train, Xk_test, yk_train, yk_test = train_test_split(
                Xk, yk, test_size=0.2, random_state=42
            )

            probe_reg = RandomForestRegressor(
                n_estimators=200, min_samples_leaf=3, random_state=42
            )
            probe_reg.fit(Xk_train, yk_train)
            package_mae = mean_absolute_error(
                yk_test, probe_reg.predict(Xk_test)
            )

            package_model = RandomForestRegressor(
                n_estimators=200, min_samples_leaf=3, random_state=42
            )
            package_model.fit(Xk, yk)

    return {
        "regressor": regressor,
        "classifier": classifier,
        "placement": placement_model,
        "package": package_model,
        "package_mae": package_mae,
        "metrics": metrics
    }


# =====================================================
# CLEAN STUDENT DATA
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

    if "Second_Internal_Marks" not in data.columns:
        data["Second_Internal_Marks"] = np.nan

    data["Second_Internal_Marks"] = pd.to_numeric(
        data["Second_Internal_Marks"],
        errors="coerce"
    ).clip(0, 100)

    for col in ["Student_ID", "Student_Name", "Gender"]:
        data[col] = data[col].fillna("").astype(str)

    for col in FEATURES:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

        data[col] = data[col].fillna(
            training_data[col].median()
        )

    for col in [
        "Attendance", "Assignment_Score",
        "Internal_Marks", "Previous_Marks"
    ]:
        data[col] = data[col].clip(0, 100)

    data["Study_Hours"] = data["Study_Hours"].clip(0, 24)

    # ----- Placement inputs (optional in the upload) -----

    imputed = pd.Series(0, index=data.index)

    for col in PLACEMENT_NUMERIC:

        if col not in data.columns:
            data[col] = np.nan

        data[col] = pd.to_numeric(data[col], errors="coerce")
        imputed += data[col].isna().astype(int)

        data[col] = data[col].fillna(training_data[col].median())

        low, high = PLACEMENT_LIMITS[col]
        data[col] = data[col].clip(low, high)

        if col in COUNT_COLUMNS or col == "Placement_Training":
            data[col] = data[col].round()

    if "Branch" not in data.columns:
        data["Branch"] = np.nan

    lookup = {b.lower(): b for b in BRANCHES}

    mapped = (
        data["Branch"].astype(str).str.strip().str.lower().map(lookup)
    )

    imputed += mapped.isna().astype(int)

    data["Branch"] = mapped.fillna(
        training_data["Branch"].mode().iloc[0]
    )

    data["Imputed_Fields"] = imputed

    return data


# =====================================================
# PREDICTIONS
# =====================================================

def predict_students(data, models):

    regressor = models["regressor"]
    classifier = models["classifier"]

    X = data[FEATURES]

    marks = np.clip(regressor.predict(X), 0, 100)
    predicted_result = classifier.predict(X)

    pass_probability = positive_probability(classifier, X) * 100

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

    # ----- Placement predictions -----

    Xp = build_placement_matrix(output)
    placement_probability = positive_probability(
        models["placement"], Xp
    ) * 100

    output["Placement_Probability"] = np.round(placement_probability, 1)

    output["Placement_Prediction"] = np.where(
        placement_probability >= CUSP_THRESHOLD, "Placed", "Not Placed"
    )

    output["Placement_Status"] = [
        placement_tier(p)["short"] for p in placement_probability
    ]

    if models["package"] is not None:
        package = models["package"].predict(Xp)
        output["Estimated_Package_LPA"] = np.where(
            placement_probability >= CUSP_THRESHOLD,
            np.round(package, 1),
            np.nan
        )
    else:
        output["Estimated_Package_LPA"] = np.nan

    criteria_results = [
        get_placement_criteria(row) for _, row in output.iterrows()
    ]

    output["Criteria_Met"] = [
        sum(1 for item in result if item[1])
        for result in criteria_results
    ]

    output["Eligible_For_Drives"] = [
        "Yes" if all(item[1] for item in result if item[3]) else "No"
        for result in criteria_results
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
        start, start + len(new_data)
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


def get_placement_suggestions(student):

    suggestions = []

    if student["Backlogs"] > 0:
        suggestions.append(
            f"📌 Clear your {int(student['Backlogs'])} backlog(s). "
            "Most recruiters require zero active backlogs."
        )

    if student["CGPA"] < 6.0:
        suggestions.append(
            "🎓 CGPA is below the common 6.0 cut-off. "
            "Prioritise improving it."
        )
    elif student["CGPA"] < 7.0:
        suggestions.append(
            "🎓 Push your CGPA towards 7+ to qualify for more companies."
        )

    if student["Tenth_Pct"] < 60 or student["Twelfth_Pct"] < 60:
        suggestions.append(
            "📉 Some companies need 60%+ in 10th/12th. "
            "Strengthen skills and projects to compensate."
        )

    if student["Coding_Score"] < 60:
        suggestions.append(
            "💻 Practise DSA and aptitude daily; target a coding score "
            "of 60+."
        )

    if student["Communication_Score"] < 60:
        suggestions.append(
            "🗣️ Join mock interviews and group discussions to improve "
            "communication."
        )

    if student["Internships"] == 0:
        suggestions.append(
            "🏢 Apply for at least one internship. It is one of the "
            "strongest hiring signals."
        )

    if student["Projects"] < 2:
        suggestions.append(
            "🛠️ Build 2+ solid projects and host them on GitHub."
        )

    if student["Certifications"] == 0:
        suggestions.append(
            "📜 Earn a relevant certification (cloud, data, "
            "programming)."
        )

    if student["Hackathons"] == 0:
        suggestions.append(
            "🏁 Take part in hackathons to show teamwork under pressure."
        )

    if student["Placement_Training"] < 1:
        suggestions.append(
            "🧑‍🏫 Attend the placement training programme."
        )

    if not suggestions:
        suggestions.append(
            "🌟 Excellent placement profile! Start applying to "
            "top drives."
        )

    return suggestions


def placement_levers(student, models):
    """What-if analysis: change one thing, re-predict, report the gain."""

    base = float(student["Placement_Probability"])

    scenarios = []

    if student["Backlogs"] > 0:
        scenarios.append(("Clear all backlogs", {"Backlogs": 0}))

    scenarios += [
        ("+1 internship",
         {"Internships": student["Internships"] + 1}),
        ("+15 coding score",
         {"Coding_Score": min(student["Coding_Score"] + 15, 100)}),
        ("+15 communication score",
         {"Communication_Score":
          min(student["Communication_Score"] + 15, 100)}),
        ("+2 certifications",
         {"Certifications": student["Certifications"] + 2}),
        ("+1 project",
         {"Projects": student["Projects"] + 1}),
        ("+1 hackathon",
         {"Hackathons": student["Hackathons"] + 1}),
        ("+0.5 CGPA",
         {"CGPA": min(student["CGPA"] + 0.5, 10)})
    ]

    if student["Placement_Training"] < 1:
        scenarios.append(
            ("Attend placement training", {"Placement_Training": 1})
        )

    rows = []

    for _, changes in scenarios:
        modified = student.copy()
        for key, value in changes.items():
            modified[key] = value
        rows.append(modified)

    Xs = build_placement_matrix(pd.DataFrame(rows).reset_index(drop=True))
    new_probability = positive_probability(models["placement"], Xs) * 100

    result = pd.DataFrame({
        "Improvement": [label for label, _ in scenarios],
        "New Probability": np.round(new_probability, 1),
        "Gain": np.round(new_probability - base, 1)
    })

    return result.sort_values("Gain", ascending=False)


# =====================================================
# INDIVIDUAL DASHBOARD: PLACEMENT TAB
# =====================================================

def render_criteria_cards(criteria):

    for start in range(0, len(criteria), 4):

        columns = st.columns(4)

        for col, (label, met, value, core) in zip(
            columns, criteria[start:start + 4]
        ):
            color = "#10B981" if met else "#F43F5E"
            icon = "✅" if met else "❌"
            tag = "CORE ELIGIBILITY" if core else "SKILL BENCHMARK"

            with col:
                render_html(f"""
                <div class="criteria-card" style="border-color:{color};">
                <div style="font-size:26px;">{icon}</div>
                <div style="font-weight:700;">{label}</div>
                <div style="color:{color};font-weight:800;font-size:18px;">{value}</div>
                <div style="font-size:10px;color:#89968D;letter-spacing:1px;">{tag}</div>
                </div>
                """)


def bucket_rate_chart(training_data, student, column, cap, title, key):

    bucket = training_data[column].clip(upper=cap).astype(int)
    rates = (
        training_data.groupby(bucket)["Placed"].mean() * 100
    ).reset_index()
    rates.columns = ["Value", "Placement Rate"]

    rates["Label"] = rates["Value"].astype(str)
    rates.loc[rates["Value"] == cap, "Label"] = f"{cap}+"

    student_bucket = min(int(student[column]), cap)

    rates["Group"] = np.where(
        rates["Value"] == student_bucket, "This student", "Others"
    )

    fig = px.bar(
        rates,
        x="Label",
        y="Placement Rate",
        color="Group",
        text=rates["Placement Rate"].round(0).astype(int).astype(str) + "%",
        title=title,
        color_discrete_map={
            "This student": "#315FAD",
            "Others": "#A7D3B7"
        }
    )

    fig.update_layout(
        yaxis_range=[0, 100],
        yaxis_title="Placement rate (%)",
        xaxis_title=column.replace("_", " "),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True, key=key)


def render_placement_tab(student, training_data, models):

    probability = float(student["Placement_Probability"])
    tier = placement_tier(probability)

    # ----- Eye-catching status banner -----

    render_html(f"""
    <div class="placement-banner" style="background:linear-gradient(135deg,{tier['c1']},{tier['c2']});">
    <div class="pb-emoji">{tier['emoji']}</div>
    <div class="pb-title">{tier['label']}</div>
    <div class="pb-sub">{tier['message']}</div>
    <div class="pb-prob">{probability:.0f}% placement probability</div>
    </div>
    """)

    # ----- SMS alert for students needing placement support -----

    if probability < CUSP_THRESHOLD:
        st.markdown("---")
        st.markdown("## 📩 Send Placement Support SMS")
        st.warning(
            f"{student['Student_Name']} has a placement probability below "
            f"{CUSP_THRESHOLD}%. You can send a support alert."
        )

        sms_phone = st.text_input(
            "Recipient mobile number (with country code)",
            value="+919632919123",
            key=f"sms_phone_{student['Student_ID']}"
        )

        sms_message = (
            f"Placement Support Alert: {student['Student_Name']} "
            f"({student['Branch']}) has been flagged as needing placement "
            "preparation. Please review the placement dashboard and "
            "contact the placement coordinator for support."
        )

        st.text_area(
            "SMS message",
            value=sms_message,
            height=110,
            key=f"sms_message_{student['Student_ID']}",
            disabled=True
        )

        if st.button(
            "📨 Send SMS Alert",
            key=f"send_sms_{student['Student_ID']}",
            use_container_width=True
        ):
            if not sms_phone.strip().startswith("+") or len(sms_phone.strip()) < 10:
                st.error(
                    "Enter a valid phone number with country code, "
                    "for example +919632919123."
                )
            else:
                try:
                    with st.spinner("Sending SMS through Twilio..."):
                        sms_sid = send_sms(sms_phone.strip(), sms_message)
                    st.success("SMS request submitted successfully to Twilio.")
                    st.caption(f"Twilio Message SID: {sms_sid}")
                except Exception as error:
                    st.error(f"SMS could not be sent: {error}")
                    st.info(
                        "Check your Twilio Account SID, Auth Token and "
                        "Twilio phone number in Streamlit Secrets. Trial "
                        "accounts may require recipient verification, and "
                        "SMS delivery to India depends on your Twilio "
                        "sender/account configuration."
                    )
    else:
        st.caption(
            "📩 SMS support alert is available when placement probability "
            f"is below {CUSP_THRESHOLD}%."
        )

    criteria = get_placement_criteria(student)
    core_met = all(item[1] for item in criteria if item[3])
    met_count = sum(1 for item in criteria if item[1])

    if core_met:
        chip = (
            '<span class="chip" style="background:#047857;">'
            '✅ ELIGIBLE FOR MOST CAMPUS DRIVES</span>'
        )
    else:
        chip = (
            '<span class="chip" style="background:#DC2626;">'
            '⛔ ELIGIBILITY GAPS FOUND</span>'
        )

    render_html(
        f'<div style="text-align:center;margin-top:10px;">{chip}'
        f'<span class="chip" style="background:#315FAD;">'
        f'📋 {met_count}/{len(criteria)} CRITERIA MET</span>'
        f'<span class="chip" style="background:#6B7C72;">'
        f'🏫 {student["Branch"]}</span></div>'
    )

    if int(student.get("Imputed_Fields", 0)) > 0:
        st.warning(
            f"{int(student['Imputed_Fields'])} placement field(s) were "
            "not provided and were filled with dataset averages, so "
            "this probability is less reliable."
        )

    # ----- Gauge + key numbers -----

    g_col, m_col = st.columns([3, 2])

    with g_col:

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            number={"suffix": "%", "font": {"size": 48}},
            title={"text": "Placement Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": tier["c1"], "thickness": 0.35},
                "steps": [
                    {"range": [0, CUSP_THRESHOLD], "color": "#FDE2E2"},
                    {"range": [CUSP_THRESHOLD, READY_THRESHOLD],
                     "color": "#FDECC8"},
                    {"range": [READY_THRESHOLD, 100], "color": "#D9F2E3"}
                ]
            }
        ))

        gauge.update_layout(
            paper_bgcolor="white",
            height=320,
            margin=dict(t=70, b=10, l=30, r=30)
        )

        st.plotly_chart(gauge, use_container_width=True, key="gauge")

    with m_col:

        st.metric(
            "Predicted Outcome",
            student["Placement_Prediction"]
        )

        package = student.get("Estimated_Package_LPA", np.nan)

        if pd.isna(package):
            st.metric("Estimated Package", "—")
            st.caption(
                "Estimated package is shown when placement "
                f"probability is {CUSP_THRESHOLD}% or higher."
            )
        else:
            st.metric("Estimated Package", f"₹{package:.1f} LPA")

            if models["package_mae"] is not None:
                st.caption(
                    f"Typical error about ±₹{models['package_mae']:.1f} "
                    "LPA. This is an estimate, not an offer."
                )

        st.metric("Criteria Met", f"{met_count}/{len(criteria)}")

    # ----- Fundamental criteria -----

    st.markdown("---")
    st.markdown("## ✅ Fundamental Placement Criteria")

    render_criteria_cards(criteria)

    st.caption(
        "CORE ELIGIBILITY = cut-offs commonly used by recruiters "
        "(CGPA, backlogs, 10th and 12th). SKILL BENCHMARK = "
        "employability indicators."
    )

    # ----- Readiness radar -----

    st.markdown("---")
    st.markdown("## 🕸️ Placement Readiness Radar")

    placed_avg = training_data[
        training_data["Placed"] == 1
    ][PLACEMENT_NUMERIC].mean()

    not_placed_avg = training_data[
        training_data["Placed"] == 0
    ][PLACEMENT_NUMERIC].mean()

    radar = go.Figure()

    for name, values, color, opacity in [
        ("This student", readiness_scores(student), "#315FAD", 0.55),
        ("Placed average", readiness_scores(placed_avg), "#10B981", 0.30),
        ("Not placed average",
         readiness_scores(not_placed_avg), "#F43F5E", 0.25)
    ]:
        radar.add_trace(go.Scatterpolar(
            r=list(values) + [values[0]],
            theta=READINESS_LABELS + [READINESS_LABELS[0]],
            fill="toself",
            name=name,
            line_color=color,
            opacity=opacity
        ))

    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Student vs Placed and Not Placed Averages",
        paper_bgcolor="white"
    )

    st.plotly_chart(radar, use_container_width=True, key="readiness_radar")

    # ----- Skill score distributions -----

    st.markdown("---")
    st.markdown("## 💻 Skill Scores vs Placement Outcome")

    d1, d2 = st.columns(2)

    for col, column, key in [
        (d1, "Coding_Score", "hist_coding"),
        (d2, "Communication_Score", "hist_comm")
    ]:
        with col:
            fig = px.histogram(
                training_data,
                x=column,
                color="Placed_Label",
                barmode="overlay",
                nbins=20,
                opacity=0.65,
                title=f"{column.replace('_', ' ')} Distribution",
                color_discrete_map={
                    "Placed": "#65A982",
                    "Not Placed": "#E27C7C"
                }
            )

            fig.add_vline(
                x=student[column],
                line_dash="dash",
                line_color="#315FAD",
                annotation_text="This student"
            )

            fig.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white",
                legend_title_text=""
            )

            st.plotly_chart(fig, use_container_width=True, key=key)

    # ----- CGPA vs coding scatter -----

    fig = px.scatter(
        training_data,
        x="CGPA",
        y="Coding_Score",
        color="Placed_Label",
        opacity=0.55,
        title="CGPA vs Coding Score (who got placed)",
        color_discrete_map={
            "Placed": "#65A982",
            "Not Placed": "#E27C7C"
        }
    )

    fig.add_trace(go.Scatter(
        x=[student["CGPA"]],
        y=[student["Coding_Score"]],
        mode="markers",
        name="This student",
        marker=dict(size=17, color="#315FAD", symbol="star",
                    line=dict(width=1, color="white"))
    ))

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text=""
    )

    st.plotly_chart(fig, use_container_width=True, key="scatter_cgpa")

    # ----- Branch and experience impact -----

    st.markdown("---")
    st.markdown("## 🏫 Branch and Experience Impact")

    branch_rates = (
        training_data.groupby("Branch")["Placed"].mean() * 100
    ).reindex(BRANCHES).reset_index()
    branch_rates.columns = ["Branch", "Placement Rate"]

    branch_rates["Group"] = np.where(
        branch_rates["Branch"] == student["Branch"],
        "This student's branch", "Other branches"
    )

    fig = px.bar(
        branch_rates,
        x="Branch",
        y="Placement Rate",
        color="Group",
        text=branch_rates["Placement Rate"].round(0).astype(int).astype(str) + "%",
        title="Placement Rate by Branch",
        color_discrete_map={
            "This student's branch": "#315FAD",
            "Other branches": "#A7D3B7"
        }
    )

    fig.update_layout(
        yaxis_range=[0, 100],
        yaxis_title="Placement rate (%)",
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True, key="branch_rate")

    e1, e2, e3 = st.columns(3)

    with e1:
        bucket_rate_chart(
            training_data, student, "Internships", 3,
            "Placement Rate by Internships", "bucket_internships"
        )

    with e2:
        bucket_rate_chart(
            training_data, student, "Backlogs", 3,
            "Placement Rate by Backlogs", "bucket_backlogs"
        )

    with e3:
        bucket_rate_chart(
            training_data, student, "Projects", 4,
            "Placement Rate by Projects", "bucket_projects"
        )

    # ----- What-if levers -----

    st.markdown("---")
    st.markdown("## 🎚️ What Would Raise the Placement Chances?")

    levers = placement_levers(student, models)
    best = levers.iloc[0]

    if best["Gain"] > 0.5:
        st.success(
            f"Biggest lever: **{best['Improvement']}** could lift "
            f"placement probability by about **{best['Gain']:.1f} points**."
        )

    levers["Direction"] = np.where(
        levers["Gain"] >= 0, "Improves", "No improvement"
    )

    fig = px.bar(
        levers.sort_values("Gain"),
        x="Gain",
        y="Improvement",
        orientation="h",
        color="Direction",
        text=levers.sort_values("Gain")["Gain"].map(lambda v: f"{v:+.1f}"),
        title="Estimated Change in Placement Probability (points)",
        color_discrete_map={
            "Improves": "#10B981",
            "No improvement": "#B8C4BC"
        }
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        xaxis_title="Probability change (percentage points)",
        yaxis_title=""
    )

    st.plotly_chart(fig, use_container_width=True, key="levers")

    st.caption(
        "Model-based what-if estimates. They show which factors the "
        "model rewards most, not a guarantee of results."
    )

    # ----- Placement feature importance -----

    st.markdown("---")
    st.markdown("### 🧠 Placement Model Feature Importance")

    importance = pd.Series(
        models["placement"].feature_importances_,
        index=PLACEMENT_COLUMNS
    )

    branch_cols = [f"Branch_{b}" for b in BRANCHES]
    branch_total = importance[branch_cols].sum()
    importance = importance.drop(branch_cols)
    importance["Branch"] = branch_total
    importance = importance.sort_values().tail(12)
    importance.index = [i.replace("_", " ") for i in importance.index]

    fig = px.bar(
        x=importance.values,
        y=importance.index,
        orientation="h",
        title="Top Factors Used by the Placement Model",
        color=importance.values,
        color_continuous_scale="Greens"
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        coloraxis_showscale=False,
        xaxis_title="Importance",
        yaxis_title=""
    )

    st.plotly_chart(fig, use_container_width=True, key="placement_importance")

    st.caption(
        "Overall model importance, not the exact contribution for "
        "this student."
    )

    # ----- Suggestions and summary -----

    st.markdown("---")
    st.markdown("## 💡 Placement Improvement Plan")

    for suggestion in get_placement_suggestions(student):
        st.info(suggestion)

    st.markdown("---")
    st.markdown("## 📝 Placement Summary")

    name = student["Student_Name"]

    if tier["short"] == "Placement Ready":
        st.success(
            f"{name} has a strong profile ({probability:.0f}% placement "
            "probability). Keep sharpening interview skills and apply "
            "widely."
        )
    elif tier["short"] == "Almost There":
        st.warning(
            f"{name} is close ({probability:.0f}%). Work on the "
            "highest-impact improvements above."
        )
    else:
        st.error(
            f"{name} needs focused preparation ({probability:.0f}%). "
            "Start with the core eligibility gaps and follow the plan."
        )


# =====================================================
# INDIVIDUAL DASHBOARD: ACADEMIC TAB (original sections)
# =====================================================

def render_academic_tab(student, training_data, regressor, classifier):

    # PREDICTION METRICS

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

    # SECOND INTERNAL COMPARISON

    st.markdown("---")
    st.markdown("## 📚 Internal Examination Progress")

    first_internal = float(student["Internal_Marks"])
    second_internal = student.get(
        "Second_Internal_Marks", np.nan
    )

    if pd.isna(second_internal):

        st.info(
            "Second Internal Marks have not been entered for this student. "
            "Add the Second_Internal_Marks column to the CSV or enter the "
            "student manually to view progress."
        )

    else:

        second_internal = float(second_internal)
        difference = second_internal - first_internal

        m1, m2, m3 = st.columns(3)

        m1.metric("First Internal", f"{first_internal:.1f}/100")
        m2.metric("Second Internal", f"{second_internal:.1f}/100")
        m3.metric(
            "Change in Marks",
            f"{difference:+.1f}",
            delta=f"{difference:+.1f} marks"
        )

        if difference > 0:
            st.success(
                f"📈 Improvement observed! {student['Student_Name']} "
                f"improved by {difference:.1f} marks in the second "
                "internal examination."
            )

        elif difference < 0:
            st.warning(
                f"📉 Marks have decreased by {abs(difference):.1f}. "
                f"{student['Student_Name']} should focus on the topics "
                "where performance has declined."
            )

        else:
            st.info(
                f"➡️ No change in marks. {student['Student_Name']} "
                "maintained the same score in both internal examinations."
            )

        comparison = pd.DataFrame({
            "Examination": ["First Internal", "Second Internal"],
            "Marks": [first_internal, second_internal]
        })

        fig = px.bar(
            comparison,
            x="Examination",
            y="Marks",
            color="Examination",
            text="Marks",
            title="First vs Second Internal Marks",
            color_discrete_sequence=["#65A982", "#A7B8D5"]
        )

        fig.update_layout(
            yaxis_range=[0, 100],
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    # ATTENDANCE PIE CHART

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
        annotations=[dict(
            text=f"{attendance:.1f}%",
            x=0.5, y=0.5,
            font_size=25,
            showarrow=False
        )],
        paper_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    if attendance < 75:
        st.warning("Attendance is below the 75% reference target.")
    else:
        st.success("Attendance meets the 75% reference target.")

    # REGRESSION ANALYSIS

    st.markdown("---")
    st.markdown("## 📈 Attendance vs Academic Performance")

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
        }
    )

    # Add a simple regression line
    if regression_data["Attendance"].nunique() > 1:
        x_values = regression_data["Attendance"].to_numpy()
        y_values = regression_data["Final_Marks"].to_numpy()

        slope, intercept = np.polyfit(x_values, y_values, 1)

        line_x = np.linspace(x_values.min(), x_values.max(), 100)
        line_y = slope * line_x + intercept

        fig.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="Regression Line",
            line=dict(color="#E28A55", width=3)
        ))

    fig.add_trace(go.Scatter(
        x=[student["Attendance"]],
        y=[student["Predicted_Marks"]],
        mode="markers",
        name="Student Prediction",
        marker=dict(
            size=15,
            color="#315FAD",
            symbol="star"
        )
    ))

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The regression line shows the overall trend in the training data. "
        "The star represents this student's predicted marks."
    )

    # RADAR CHART

    st.markdown("---")
    st.markdown("## 🕸️ Student Performance Radar")

    radar_labels = [
        "Attendance",
        "Assignments",
        "Internal Marks",
        "Previous Marks",
        "Study Hours"
    ]

    radar_values = [
        student["Attendance"],
        student["Assignment_Score"],
        student["Internal_Marks"],
        student["Previous_Marks"],
        min(student["Study_Hours"] / 8 * 100, 100)
    ]

    average_values = [
        training_data["Attendance"].mean(),
        training_data["Assignment_Score"].mean(),
        training_data["Internal_Marks"].mean(),
        training_data["Previous_Marks"].mean(),
        min(training_data["Study_Hours"].mean() / 8 * 100, 100)
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=radar_values + [radar_values[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        name="Student",
        line_color="#65A982"
    ))

    fig.add_trace(go.Scatterpolar(
        r=average_values + [average_values[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        name="Training Average",
        line_color="#A7B8D5",
        opacity=0.65
    ))

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

    # FACTOR COMPARISON

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
        color_discrete_sequence=["#65A982", "#A7B8D5"]
    )

    st.plotly_chart(fig, use_container_width=True)

    # MARKS DISTRIBUTION

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

    # MODEL FEATURE IMPORTANCE

    st.markdown("---")
    st.markdown("### 🧠 Model Feature Importance")

    importance_data = pd.DataFrame({
        "Feature": FEATURES * 2,
        "Importance": (
            list(regressor.feature_importances_)
            + list(classifier.feature_importances_)
        ),
        "Model": (
            ["Marks Model"] * len(FEATURES)
            + ["Pass/Fail Model"] * len(FEATURES)
        )
    })

    fig = px.bar(
        importance_data,
        x="Feature",
        y="Importance",
        color="Model",
        barmode="group",
        title="Overall Prediction Model Criteria",
        color_discrete_sequence=["#65A982", "#A7B8D5"]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This shows overall model feature importance, not the exact "
        "contribution of each factor to this student's prediction."
    )

    # SUGGESTIONS

    st.markdown("---")
    st.markdown("## 💡 Personalized Academic Suggestions")

    for suggestion in get_suggestions(student):
        st.info(suggestion)

    # SUMMARY

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
            "Focus on weaker criteria and follow the suggestions."
        )


# =====================================================
# INDIVIDUAL STUDENT DASHBOARD
# =====================================================

def show_individual_dashboard(
    student, training_data, models
):

    if st.button("← Back to Main Dashboard"):
        st.query_params.clear()
        st.rerun()

    st.markdown("---")

    tier = placement_tier(float(student["Placement_Probability"]))

    render_html(f"""
    <div class="student-card">
    <div style="font-size:75px;">{student['Avatar']}</div>
    <h1 style="color:#294C3A;">{student['Student_Name']}</h1>
    <p>Student ID: {student['Student_ID']}</p>
    <p>Gender: {student['Gender']} &nbsp;|&nbsp; Branch: {student['Branch']}</p>
    <p>Personal Academic &amp; Placement Dashboard</p>
    <span class="pill" style="background:linear-gradient(135deg,{tier['c1']},{tier['c2']});">{tier['emoji']} {tier['short']}</span>
    </div>
    """)

    st.markdown("")

    tab_placement, tab_academic = st.tabs([
        "🎯 Placement Insights",
        "📚 Academic Insights"
    ])

    with tab_placement:
        render_placement_tab(student, training_data, models)

    with tab_academic:
        render_academic_tab(
            student,
            training_data,
            models["regressor"],
            models["classifier"]
        )


# =====================================================
# COMPANY HIRING SIMULATOR
# =====================================================

# NOTE: these cut-offs and weights are typical-practice assumptions,
# not facts about any real company. Edit them to match your campus.

COMPANY_PROFILES = {
    "General recruiter": {
        "icon": "🏢",
        "about": "Balanced screening used by most recruiters.",
        "min_cgpa": 6.0, "max_backlogs": 0,
        "min_tenth": 60, "min_twelfth": 60, "min_coding": 0,
        "branches": BRANCHES,
        "weights": {
            "CGPA": 25, "Coding": 25, "Communication": 20,
            "Projects": 10, "Internships": 10,
            "Certifications": 5, "Hackathons": 5
        }
    },
    "Product company": {
        "icon": "🚀",
        "about": "High coding bar, strong CGPA, CS/IT/ECE branches.",
        "min_cgpa": 7.0, "max_backlogs": 0,
        "min_tenth": 60, "min_twelfth": 60, "min_coding": 65,
        "branches": ["CSE", "IT", "ECE"],
        "weights": {
            "Coding": 35, "CGPA": 20, "Projects": 20,
            "Internships": 10, "Languages": 10, "Hackathons": 5
        }
    },
    "Service company": {
        "icon": "💼",
        "about": "Mass hiring: CGPA, aptitude and communication matter "
                 "most; all branches welcome.",
        "min_cgpa": 6.0, "max_backlogs": 0,
        "min_tenth": 60, "min_twelfth": 60, "min_coding": 40,
        "branches": BRANCHES,
        "weights": {
            "Communication": 30, "CGPA": 25, "Coding": 20,
            "Attendance": 10, "Certifications": 10, "Internships": 5
        }
    },
    "Core company": {
        "icon": "⚙️",
        "about": "Core engineering roles: strong CGPA, internships "
                 "and projects in Mechanical/Civil/EEE/ECE.",
        "min_cgpa": 6.5, "max_backlogs": 0,
        "min_tenth": 60, "min_twelfth": 60, "min_coding": 0,
        "branches": ["Mechanical", "Civil", "EEE", "ECE"],
        "weights": {
            "CGPA": 40, "Internships": 20, "Projects": 20,
            "Communication": 10, "Certifications": 10
        }
    },
    "Startup": {
        "icon": "⚡",
        "about": "Flexible on marks, strong on projects, coding and "
                 "hands-on experience.",
        "min_cgpa": 6.0, "max_backlogs": 1,
        "min_tenth": 50, "min_twelfth": 50, "min_coding": 55,
        "branches": BRANCHES,
        "weights": {
            "Projects": 25, "Coding": 25, "Internships": 20,
            "Hackathons": 15, "Communication": 10, "Languages": 5
        }
    }
}


def axis_frame(df):
    """Skill axes on a 0-100 scale for every student."""
    return pd.DataFrame({
        "CGPA": (df["CGPA"].astype(float) * 10).clip(upper=100),
        "Coding": df["Coding_Score"].astype(float),
        "Communication": df["Communication_Score"].astype(float),
        "Projects": (df["Projects"].astype(float) / 5 * 100).clip(upper=100),
        "Internships": (df["Internships"].astype(float) / 3 * 100).clip(upper=100),
        "Certifications": (df["Certifications"].astype(float) / 4 * 100).clip(upper=100),
        "Hackathons": (df["Hackathons"].astype(float) / 3 * 100).clip(upper=100),
        "Languages": (df["Programming_Languages"].astype(float) / 5 * 100).clip(upper=100),
        "Attendance": df["Attendance"].astype(float)
    })


def rank_candidates(results, weights, rules, vacancies, waitlist):
    """Screen, score and rank students for one company.

    Hire score = 40% model placement probability
               + 60% company-specific fit (weighted skills)."""

    sim = results.copy()

    for col in [
        "CGPA", "Backlogs", "Tenth_Pct", "Twelfth_Pct",
        "Coding_Score", "Placement_Probability"
    ]:
        sim[col] = pd.to_numeric(sim[col], errors="coerce")

    axes = axis_frame(sim)

    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    fit = (axes[w.index] * w).sum(axis=1)

    sim["Company_Fit"] = fit.round(1)
    sim["Hire_Score"] = (
        0.4 * sim["Placement_Probability"] + 0.6 * fit
    ).round(1)

    checks = [
        (sim["CGPA"] >= rules["min_cgpa"],
         f"CGPA below {rules['min_cgpa']:.1f}"),
        (sim["Backlogs"] <= rules["max_backlogs"],
         f"more than {int(rules['max_backlogs'])} backlog(s)"),
        (sim["Tenth_Pct"] >= rules["min_tenth"],
         f"10th below {rules['min_tenth']:.0f}%"),
        (sim["Twelfth_Pct"] >= rules["min_twelfth"],
         f"12th below {rules['min_twelfth']:.0f}%"),
        (sim["Coding_Score"] >= rules["min_coding"],
         f"coding score below {rules['min_coding']:.0f}"),
        (sim["Branch"].isin(rules["branches"]),
         "branch not hired by this company")
    ]

    eligible = pd.Series(True, index=sim.index)
    reasons = pd.Series("", index=sim.index)

    for passed, text in checks:
        eligible &= passed
        reasons = reasons.where(passed, reasons + text + "; ")

    sim["Eligible"] = eligible
    sim["Reason"] = np.where(
        eligible,
        "Eligible, ranked below the cut-off",
        reasons.str.rstrip("; ")
    )

    order = sim[eligible].sort_values(
        ["Hire_Score", "CGPA"], ascending=False
    ).index

    sim["Rank"] = np.nan
    sim.loc[order, "Rank"] = np.arange(1, len(order) + 1)

    sim["Outcome"] = "Not selected"
    sim.loc[~eligible, "Outcome"] = "Not eligible"
    sim.loc[order[:vacancies], "Outcome"] = "Selected"
    sim.loc[order[vacancies:vacancies + waitlist], "Outcome"] = "Waitlist"

    sim.loc[order[:vacancies], "Reason"] = "Selected"
    sim.loc[order[vacancies:vacancies + waitlist], "Reason"] = (
        "Waitlisted: next in line"
    )

    return sim


HIRING_COLUMNS = [
    "Rank", "Student_ID", "Student_Name", "Branch", "CGPA", "Backlogs",
    "Coding_Score", "Communication_Score", "Projects", "Internships",
    "Placement_Probability", "Company_Fit", "Hire_Score"
]


def hiring_table(df, with_reason=False):
    cols = HIRING_COLUMNS + (["Reason"] if with_reason else [])
    table = df[cols].copy()
    table["Rank"] = table["Rank"].astype("Int64")

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hire_Score": st.column_config.ProgressColumn(
                "Hire Score", min_value=0, max_value=100, format="%.1f"
            ),
            "Placement_Probability": st.column_config.NumberColumn(
                "Placement Prob (%)", format="%.1f"
            ),
            "Company_Fit": st.column_config.NumberColumn(
                "Company Fit", format="%.1f"
            )
        }
    )


def render_hiring_simulator(results):

    st.markdown("---")
    st.markdown("## 🏢 Company Hiring Simulator")

    st.caption(
        "Pick a company type and the number of vacancies to see which "
        "students are most likely to be selected. Students are screened "
        "on the company's criteria, then ranked by a hire score "
        "(40% model placement probability + 60% company-specific fit)."
    )

    total = len(results)

    c1, c2, c3 = st.columns([2, 1, 1])

    company = c1.selectbox(
        "Company type",
        list(COMPANY_PROFILES),
        key="hire_company"
    )

    vacancies = int(c2.number_input(
        "Vacancies (students to hire)",
        min_value=1,
        max_value=max(total, 1),
        value=min(20, total),
        step=1,
        key="hire_vacancies"
    ))

    waitlist = int(c3.number_input(
        "Waitlist size",
        min_value=0,
        max_value=100,
        value=5,
        step=1,
        key="hire_waitlist"
    ))

    profile = COMPANY_PROFILES[company]
    tag = company.replace(" ", "_")

    st.info(f"{profile['icon']} **{company}:** {profile['about']}")

    with st.expander(
        "⚙️ Screening criteria (pre-filled for this company type, editable)"
    ):
        a1, a2, a3 = st.columns(3)

        min_cgpa = a1.number_input(
            "Minimum CGPA", 0.0, 10.0, float(profile["min_cgpa"]),
            step=0.1, key=f"hire_cgpa_{tag}"
        )
        max_backlogs = a2.number_input(
            "Max backlogs allowed", 0, 15, int(profile["max_backlogs"]),
            step=1, key=f"hire_backlogs_{tag}"
        )
        min_coding = a3.number_input(
            "Minimum coding score", 0, 100, int(profile["min_coding"]),
            step=5, key=f"hire_coding_{tag}"
        )

        b1, b2, b3 = st.columns(3)

        min_tenth = b1.number_input(
            "Minimum 10th %", 0, 100, int(profile["min_tenth"]),
            step=5, key=f"hire_tenth_{tag}"
        )
        min_twelfth = b2.number_input(
            "Minimum 12th %", 0, 100, int(profile["min_twelfth"]),
            step=5, key=f"hire_twelfth_{tag}"
        )
        branches = b3.multiselect(
            "Branches hired", BRANCHES,
            default=profile["branches"], key=f"hire_branches_{tag}"
        )

        st.caption(
            "Ranking weights for this company type: "
            + ", ".join(
                f"{name} {weight}%"
                for name, weight in profile["weights"].items()
            )
            + ". These are typical-practice assumptions, not real "
            "company data."
        )

    rules = {
        "min_cgpa": min_cgpa,
        "max_backlogs": max_backlogs,
        "min_tenth": min_tenth,
        "min_twelfth": min_twelfth,
        "min_coding": min_coding,
        "branches": branches
    }

    sim = rank_candidates(
        results, profile["weights"], rules, vacancies, waitlist
    )

    selected = sim[sim["Outcome"] == "Selected"].sort_values("Rank")
    waitlisted = sim[sim["Outcome"] == "Waitlist"].sort_values("Rank")
    not_selected = sim[
        ~sim["Outcome"].isin(["Selected", "Waitlist"])
    ].sort_values(
        ["Rank", "Hire_Score"], ascending=[True, False],
        na_position="last"
    )

    filled = len(selected)
    eligible_count = int(sim["Eligible"].sum())
    rate = vacancies / total * 100

    cutoff_text = (
        f"{selected['Hire_Score'].min():.1f}" if filled else "n/a"
    )

    render_html(f"""
    <div class="placement-banner" style="background:linear-gradient(135deg,#1E3A8A,#3B82F6);">
    <div class="pb-emoji">{profile['icon']}</div>
    <div class="pb-title">{filled} STUDENTS SELECTED</div>
    <div class="pb-sub">{company} hiring {vacancies} out of {total} applicants ({rate:.1f}% selection rate)</div>
    <div class="pb-prob">Cut-off hire score: {cutoff_text}</div>
    </div>
    """)

    st.markdown("")

    if filled < vacancies:
        st.warning(
            f"Only {eligible_count} student(s) meet the screening "
            f"criteria, so {vacancies - filled} vacancy(ies) stay "
            "unfilled. Relax the criteria above to widen the pool."
        )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("👥 Applicants", total)
    m2.metric("📋 Pass Screening", eligible_count)
    m3.metric("✅ Selected", filled)
    m4.metric("⏳ Waitlisted", len(waitlisted))

    tab_sel, tab_wait, tab_rest = st.tabs([
        f"✅ Selected ({filled})",
        f"⏳ Waitlist ({len(waitlisted)})",
        f"❌ Not selected ({len(not_selected)})"
    ])

    with tab_sel:
        if filled:
            hiring_table(selected)
        else:
            st.info("No student meets the screening criteria.")

    with tab_wait:
        if len(waitlisted):
            hiring_table(waitlisted)
        else:
            st.info("No waitlisted students.")

    with tab_rest:
        if len(not_selected):
            hiring_table(not_selected, with_reason=True)
        else:
            st.info("Everyone was selected or waitlisted.")

    shortlist = pd.concat([selected, waitlisted])

    if len(shortlist):
        export = shortlist[HIRING_COLUMNS + ["Outcome"]].copy()

        st.download_button(
            "⬇️ Download Selected and Waitlist",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name=f"hiring_shortlist_{tag.lower()}.csv",
            mime="text/csv"
        )

    ranked = sim[sim["Eligible"]].sort_values("Rank")

    if len(ranked):

        ch1, ch2 = st.columns(2)

        with ch1:
            fig = px.scatter(
                ranked,
                x="Rank",
                y="Hire_Score",
                color="Outcome",
                title="Hire Score by Rank (cut-off line)",
                color_discrete_map={
                    "Selected": "#10B981",
                    "Waitlist": "#F59E0B",
                    "Not selected": "#B8C4BC"
                },
                hover_data=["Student_Name", "Branch"]
            )

            fig.add_vline(
                x=vacancies + 0.5,
                line_dash="dash",
                line_color="#1E3A8A",
                annotation_text="Hiring cut-off"
            )

            fig.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white",
                yaxis_title="Hire score",
                legend_title_text=""
            )

            st.plotly_chart(
                fig, use_container_width=True, key="hire_rank_chart"
            )

        with ch2:
            if filled:
                by_branch = (
                    selected["Branch"].value_counts()
                    .reindex(BRANCHES, fill_value=0).reset_index()
                )
                by_branch.columns = ["Branch", "Selected"]

                fig = px.bar(
                    by_branch,
                    x="Branch",
                    y="Selected",
                    text="Selected",
                    title="Selected Students by Branch",
                    color_discrete_sequence=["#3B82F6"]
                )

                fig.update_layout(
                    paper_bgcolor="white",
                    plot_bgcolor="white"
                )

                st.plotly_chart(
                    fig, use_container_width=True,
                    key="hire_branch_chart"
                )

    st.caption(
        "This is a ranking simulation based on the model and assumed "
        "company criteria. Real selection also depends on interviews, "
        "openings and other factors, so treat it as guidance, not a "
        "guarantee."
    )


# =====================================================
# COHORT HEATMAP (single heatmap on the main page)
# =====================================================

def render_cohort_heatmap(results):

    st.markdown("---")
    st.markdown("## 🔥 Cohort Criteria Heatmap")

    h1, h2 = st.columns([2, 1])

    mode = h1.selectbox(
        "Students to show",
        [
            "Highest placement probability",
            "Lowest placement probability",
            "Upload order"
        ],
        key="heatmap_mode"
    )

    limit = min(len(results), 100)

    count = int(h2.number_input(
        "Number of students",
        min_value=1,
        max_value=limit,
        value=min(20, limit),
        step=1,
        key="heatmap_count"
    ))

    if mode == "Highest placement probability":
        view = results.sort_values(
            "Placement_Probability", ascending=False
        ).head(count)
    elif mode == "Lowest placement probability":
        view = results.sort_values(
            "Placement_Probability", ascending=True
        ).head(count)
    else:
        view = results.head(count)

    grid = pd.DataFrame({
        "Attendance": view["Attendance"],
        "Assignment": view["Assignment_Score"],
        "Internal": view["Internal_Marks"],
        "Previous": view["Previous_Marks"],
        "Study Hrs": (view["Study_Hours"].astype(float) / 8 * 100).clip(upper=100),
        "CGPA": view["CGPA"].astype(float) * 10,
        "10th %": view["Tenth_Pct"],
        "12th %": view["Twelfth_Pct"],
        "Coding": view["Coding_Score"],
        "Communication": view["Communication_Score"],
        "Projects": (view["Projects"].astype(float) / 5 * 100).clip(upper=100),
        "Internships": (view["Internships"].astype(float) / 3 * 100).clip(upper=100),
        "Certifications": (view["Certifications"].astype(float) / 4 * 100).clip(upper=100),
        "Hackathons": (view["Hackathons"].astype(float) / 3 * 100).clip(upper=100),
        "Languages": (view["Programming_Languages"].astype(float) / 5 * 100).clip(upper=100)
    }).astype(float)

    labels = (
        view["Student_ID"].astype(str) + " · "
        + view["Student_Name"].astype(str)
    )
    duplicated = labels.duplicated(keep=False)
    labels = labels.where(
        ~duplicated, labels + " #" + view["Record_ID"].astype(str)
    )
    grid.index = labels.values

    fig = px.imshow(
        grid,
        text_auto=".0f" if len(grid) <= 30 else False,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=100,
        title="Student Criteria Heatmap (0–100 scale)"
    )

    fig.update_yaxes(dtick=1)
    fig.update_layout(
        height=max(380, 26 * len(grid) + 150),
        paper_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True, key="cohort_heatmap")

    st.caption(
        "Green is strong, red is weak. CGPA is shown as CGPA × 10; "
        "study hours are scaled to 8 hours = 100; project, internship, "
        "certification, hackathon and language counts are scaled to a "
        "typical maximum."
    )


# =====================================================
# LOAD DATA AND TRAIN
# =====================================================

try:
    training_data = load_training_data()
    models = train_models(training_data)

except Exception as error:
    st.error(f"Training Error: {error}")
    st.stop()


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<h1 style="color:#294C3A;">🎓 Student Performance & Placement Insights</h1>',
    unsafe_allow_html=True
)

st.markdown(
    "Academic Analytics | Placement Prediction | Machine Learning"
)

with st.expander("🧪 Model and dataset information"):

    i1, i2, i3 = st.columns(3)

    i1.metric("Training students", len(training_data))
    i2.metric(
        "Placement rate in data",
        f"{training_data['Placed'].mean() * 100:.1f}%"
    )
    i3.metric(
        "Average CGPA",
        f"{training_data['CGPA'].mean():.2f}"
    )

    if models["metrics"]:
        m = models["metrics"]

        p1, p2, p3 = st.columns(3)
        p1.metric("Placement accuracy (hold-out)", f"{m['Accuracy'] * 100:.1f}%")
        p2.metric("Placement F1 score", f"{m['F1 score']:.2f}")
        p3.metric("Placement ROC-AUC", f"{m['ROC-AUC']:.2f}")

        st.caption(
            f"Measured on {m['Test rows']} students the model had not "
            "seen during that test. Probabilities are estimates, not "
            "guarantees."
        )

# =====================================================
# ROUTE TO INDIVIDUAL PAGE
# =====================================================

selected_record = st.query_params.get("student")

if selected_record is not None:

    students = st.session_state.students

    if not students.empty:

        match = students[
            students["Record_ID"].astype(str) == str(selected_record)
        ]

        if not match.empty:

            show_individual_dashboard(
                match.iloc[0],
                training_data,
                models
            )

            st.stop()

    st.warning("Student profile not found.")

    if st.button("Return to Dashboard"):
        st.query_params.clear()
        st.rerun()

    st.stop()


# =====================================================
# BULK CSV UPLOAD
# =====================================================

st.markdown("---")
st.markdown("## 📁 Bulk Student CSV Upload")

st.caption(
    "Required columns: " + ", ".join(FEATURES) + ". "
    "For accurate placement predictions also include: Branch, "
    + ", ".join(PLACEMENT_NUMERIC) + ". "
    "Missing placement columns are filled with dataset averages."
)

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

        missing_placement = [
            col for col in PLACEMENT_NUMERIC + ["Branch"]
            if col not in uploaded_data.columns
        ]

        if missing_placement:
            st.warning(
                "Placement columns not found: "
                + ", ".join(missing_placement)
                + ". Dataset averages will be used for them, so "
                "placement predictions will be less reliable."
            )

        with st.expander("Preview CSV"):
            st.dataframe(uploaded_data.head(10))

        if st.button("🚀 Predict All Students", type="primary"):

            if signature in st.session_state.processed_uploads:
                st.warning("This CSV has already been processed.")
            else:

                cleaned = clean_student_data(
                    uploaded_data,
                    training_data
                )

                predictions = predict_students(
                    cleaned,
                    models
                )

                predictions["Source"] = "Bulk CSV"

                add_students(predictions)

                st.session_state.processed_uploads.add(signature)

                st.success("Predictions completed!")

    except Exception as error:
        st.error(f"CSV Error: {error}")


# =====================================================
# SINGLE STUDENT INPUT
# =====================================================

st.markdown("---")
st.markdown("## 👤 Add Individual Student")

with st.form("single_student_form"):

    student_name = st.text_input("Student Name (optional)")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Not specified"]
    )

    student_id = st.text_input("Student ID (optional)")

    st.markdown("#### 📚 Academic Details")

    c1, c2 = st.columns(2)

    with c1:
        attendance = st.number_input(
            "Attendance (%)", 0.0, 100.0, 80.0
        )

        assignment = st.number_input(
            "Assignment Score", 0.0, 100.0, 70.0
        )

        internal = st.number_input(
            "First Internal Marks", 0.0, 100.0, 65.0
        )

        second_internal = st.number_input(
            "Second Internal Marks", 0.0, 100.0, 65.0
        )

    with c2:
        previous = st.number_input(
            "Previous Marks", 0.0, 100.0, 70.0
        )

        study_hours = st.number_input(
            "Study Hours Per Day", 0.0, 24.0, 3.0
        )

    st.markdown("#### 🎯 Placement Profile")

    p1, p2, p3 = st.columns(3)

    with p1:
        branch = st.selectbox("Branch", BRANCHES)

        cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0, step=0.1)

        backlogs = st.number_input(
            "Active Backlogs", 0, 15, 0, step=1
        )

        tenth = st.number_input(
            "10th Percentage", 0.0, 100.0, 75.0
        )

        twelfth = st.number_input(
            "12th Percentage", 0.0, 100.0, 72.0
        )

    with p2:
        coding = st.number_input(
            "Coding / Aptitude Score", 0.0, 100.0, 60.0
        )

        communication = st.number_input(
            "Communication Score", 0.0, 100.0, 60.0
        )

        languages = st.number_input(
            "Programming Languages Known", 0, 15, 2, step=1
        )

        certifications = st.number_input(
            "Certifications", 0, 20, 1, step=1
        )

    with p3:
        projects = st.number_input("Projects", 0, 20, 2, step=1)

        internships = st.number_input(
            "Internships", 0, 10, 1, step=1
        )

        hackathons = st.number_input(
            "Hackathons", 0, 15, 0, step=1
        )

        leadership = st.number_input(
            "Leadership Roles", 0, 10, 0, step=1
        )

        extracurricular = st.number_input(
            "Extracurricular Activities", 0, 15, 2, step=1
        )

    training_attended = st.checkbox("Attended placement training")

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
            "Second_Internal_Marks": second_internal,
            "Previous_Marks": previous,
            "Study_Hours": study_hours,
            "Branch": branch,
            "CGPA": cgpa,
            "Backlogs": backlogs,
            "Tenth_Pct": tenth,
            "Twelfth_Pct": twelfth,
            "Coding_Score": coding,
            "Communication_Score": communication,
            "Programming_Languages": languages,
            "Certifications": certifications,
            "Projects": projects,
            "Internships": internships,
            "Hackathons": hackathons,
            "Leadership_Roles": leadership,
            "Extracurricular_Activities": extracurricular,
            "Placement_Training": int(training_attended)
        }])

        cleaned_single = clean_student_data(single_data, training_data)

        prediction = predict_students(cleaned_single, models)

        prediction["Source"] = "Single Entry"

        add_students(prediction)

        st.success("Individual prediction completed!")


# =====================================================
# MAIN ANALYTICS
# =====================================================

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

    # ----- Placement analytics -----

    st.markdown("---")
    st.markdown("## 🎯 Overall Placement Analytics")

    placed_count = (
        results["Placement_Prediction"] == "Placed"
    ).sum()

    ready_count = (
        results["Placement_Status"] == "Placement Ready"
    ).sum()

    eligible_count = (
        results["Eligible_For_Drives"] == "Yes"
    ).sum()

    pc1, pc2, pc3, pc4 = st.columns(4)

    pc1.metric("🏆 Placement Ready", int(ready_count))
    pc2.metric(
        "💼 Predicted Placed",
        f"{int(placed_count)} ({placed_count / total * 100:.0f}%)"
    )
    pc3.metric(
        "📋 Eligible for Drives",
        f"{int(eligible_count)} ({eligible_count / total * 100:.0f}%)"
    )
    pc4.metric(
        "🎯 Avg Placement Probability",
        f"{results['Placement_Probability'].mean():.1f}%"
    )

    pl1, pl2 = st.columns(2)

    with pl1:

        status_counts = results["Placement_Status"].value_counts()

        fig = px.pie(
            names=status_counts.index,
            values=status_counts.values,
            hole=0.55,
            title="Placement Status Breakdown",
            color=status_counts.index,
            color_discrete_map=STATUS_COLORS
        )

        st.plotly_chart(
            fig, use_container_width=True, key="cohort_status_pie"
        )

    with pl2:

        fig = px.histogram(
            results,
            x="Placement_Probability",
            nbins=15,
            title="Placement Probability Distribution",
            color_discrete_sequence=["#10B981"]
        )

        st.plotly_chart(
            fig, use_container_width=True, key="cohort_prob_hist"
        )

    pl3, pl4 = st.columns(2)

    with pl3:

        branch_avg = (
            results.groupby("Branch")["Placement_Probability"]
            .mean().reset_index()
        )

        fig = px.bar(
            branch_avg,
            x="Branch",
            y="Placement_Probability",
            text=branch_avg["Placement_Probability"].round(0).astype(int).astype(str) + "%",
            title="Average Placement Probability by Branch",
            color_discrete_sequence=["#65A982"]
        )

        fig.update_layout(
            yaxis_range=[0, 100],
            yaxis_title="Probability (%)"
        )

        st.plotly_chart(
            fig, use_container_width=True, key="cohort_branch_bar"
        )

    with pl4:

        cohort_scores = np.mean(
            [readiness_scores(row) for _, row in results.iterrows()],
            axis=0
        )

        benchmark = readiness_scores(
            training_data[
                training_data["Placed"] == 1
            ][PLACEMENT_NUMERIC].mean()
        )

        gap = pd.DataFrame({
            "Skill": READINESS_LABELS * 2,
            "Score": list(cohort_scores) + list(benchmark),
            "Group": ["This cohort"] * len(READINESS_LABELS)
                     + ["Placed benchmark"] * len(READINESS_LABELS)
        })

        fig = px.bar(
            gap,
            x="Skill",
            y="Score",
            color="Group",
            barmode="group",
            title="Cohort Skills vs Placed-Student Benchmark",
            color_discrete_map={
                "This cohort": "#315FAD",
                "Placed benchmark": "#10B981"
            }
        )

        fig.update_layout(yaxis_range=[0, 100])

        st.plotly_chart(
            fig, use_container_width=True, key="cohort_skill_gap"
        )

    render_cohort_heatmap(results)

    st.download_button(
        "⬇️ Download Prediction Results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="student_prediction_results.csv",
        mime="text/csv"
    )

    # STUDENT PROFILES

    st.markdown("---")
    st.markdown("## 👥 Individual Student Profiles")

    search = st.text_input("🔍 Search student name or ID")

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

        total_pages = max(1, -(-len(filtered) // PAGE_SIZE))
        page = 1

        if total_pages > 1:
            page = int(st.number_input(
                f"Page (1 to {total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1
            ))

        first = (page - 1) * PAGE_SIZE
        page_data = filtered.iloc[first:first + PAGE_SIZE]

        st.caption(
            f"Showing {first + 1} to {first + len(page_data)} "
            f"of {len(filtered)} students"
        )

        for start in range(0, len(page_data), 3):

            columns = st.columns(3)

            for col, (_, student) in zip(
                columns,
                page_data.iloc[start:start + 3].iterrows()
            ):

                with col:

                    status_color = (
                        "#218653"
                        if student["Status"] == "On Track"
                        else "#D14D4D"
                    )

                    tier = placement_tier(
                        float(student["Placement_Probability"])
                    )

                    render_html(f"""
                    <div class="student-card">
                    <div style="font-size:55px;">{student['Avatar']}</div>
                    <h3>{student['Student_Name']}</h3>
                    <p>ID: {student['Student_ID']} &nbsp;|&nbsp; {student['Branch']}</p>
                    <p>Predicted Marks: <b>{student['Predicted_Marks']}</b></p>
                    <p>Pass Probability: <b>{student['Pass_Probability']}%</b></p>
                    <p style="color:{status_color};font-weight:bold;">{student['Status']}</p>
                    <span class="pill" style="background:linear-gradient(135deg,{tier['c1']},{tier['c2']});">{tier['emoji']} {tier['short']} · {student['Placement_Probability']:.0f}%</span>
                    </div>
                    """)

                    if st.button(
                        f"{student['Avatar']} Open Personal Dashboard",
                        key=f"student_{student['Record_ID']}",
                        use_container_width=True
                    ):

                        st.query_params["student"] = str(
                            student["Record_ID"]
                        )

                        st.rerun()

    render_hiring_simulator(results)

else:

    st.info(
        "Upload a CSV or enter a student's details to begin."
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#89968D;">
        Student Performance & Placement Insights<br>
        Data Science | Machine Learning | Academic &amp; Placement Analytics
    </div>
    """,
    unsafe_allow_html=True
)
