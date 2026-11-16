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
