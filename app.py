
def show_student_page(student, training_data, regressor, classifier):

    # BACK BUTTON
    if st.button("← Back to Main Dashboard"):
        st.query_params.clear()
        st.rerun()

    st.markdown("---")

    # STUDENT PROFILE
    st.markdown(
        f"""
        <div class="student-card">
            <div style="font-size:70px;">{student['Avatar']}</div>
            <h1 style="color:#294C3A;">{student['Student_Name']}</h1>
            <p>Student ID: {student['Student_ID']}</p>
            <p>Gender: {student['Gender']}</p>
            <p>Individual Academic Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # PREDICTION METRICS
    st.markdown("## 🎯 Academic Prediction")

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
        st.success("Student is predicted to be On Track.")
    else:
        st.warning("Student needs additional academic attention.")

    # =================================================
    # ATTENDANCE PIE CHART
    # =================================================

    st.markdown("---")
    st.markdown("## 📅 Attendance Analysis")

    attendance = float(student["Attendance"])

    attendance_data = pd.DataFrame({
        "Category": ["Present", "Remaining"],
        "Percentage": [
            attendance,
            100 - attendance
        ]
    })

    fig = px.pie(
        attendance_data,
        names="Category",
        values="Percentage",
        hole=0.55,
        title="Individual Attendance Percentage",
        color="Category",
        color_discrete_map={
            "Present": "#65A982",
            "Remaining": "#E5E9E5"
        }
    )

    fig.update_traces(
        textinfo="label+percent",
        textposition="inside"
    )

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

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if attendance < 75:
        st.warning("Attendance is below the 75% target.")
    else:
        st.success("Attendance meets the 75% target.")

    # =================================================
    # PERFORMANCE CRITERIA
    # =================================================

    st.markdown("---")
    st.markdown("## 📊 Individual Performance Criteria")

    criteria = pd.DataFrame({
        "Factor": [
            "Attendance",
            "Assignment Score",
            "Internal Marks",
            "Previous Marks",
            "Study Hours"
        ],
        "Student Value": [
            student["Attendance"],
            student["Assignment_Score"],
            student["Internal_Marks"],
            student["Previous_Marks"],
            student["Study_Hours"] / 24 * 100
        ],
        "Reference Target": [
            75, 50, 50, 50, 2 / 24 * 100
        ]
    })

    criteria_long = criteria.melt(
        id_vars="Factor",
        var_name="Category",
        value_name="Value"
    )

    fig = px.bar(
        criteria_long,
        x="Factor",
        y="Value",
        color="Category",
        barmode="group",
        title="Student Values and Reference Targets",
        color_discrete_sequence=[
            "#65A982",
            "#A7B8D5"
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =================================================
    # PREDICTION HEATMAP
    # =================================================

    st.markdown("---")
    st.markdown("## 🔥 Prediction Criteria Heatmap")

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Marks Prediction": regressor.feature_importances_,
        "Pass/Fail Prediction": classifier.feature_importances_
    }).set_index("Feature").T

    fig = px.imshow(
        importance,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="Greens",
        title="Machine Learning Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "This heatmap explains the overall importance of each criterion "
        "in the trained prediction models. It is not an individual "
        "contribution explanation."
    )

    # =================================================
    # INDIVIDUAL SUGGESTIONS
    # =================================================

    st.markdown("---")
    st.markdown("## 💡 Personalized Academic Suggestions")

    suggestions = get_suggestions(student)

    for suggestion in suggestions:
        st.info(suggestion)

    # =================================================
    # INDIVIDUAL SUMMARY
    # =================================================

    st.markdown("---")
    st.markdown("## 📝 Student Performance Summary")

    if student["Status"] == "On Track":

        st.success(
            f"{student['Student_Name']} is predicted to be on track. "
            "Continue maintaining good academic performance."
        )

    else:

        st.warning(
            f"{student['Student_Name']} may need additional academic support. "
            "Focus on the weaker performance criteria and follow the suggestions."
        )

