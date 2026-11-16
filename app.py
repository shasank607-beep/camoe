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
