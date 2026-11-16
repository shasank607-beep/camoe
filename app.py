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
