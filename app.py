from datetime import date, timedelta

import streamlit as st

from src.llm.planner import (
    StudyPlanRequest,
    generate_study_plan,
    modify_study_plan,
)
from src.ui.session import initialize_session


st.set_page_config(
    page_title="AI Academic Assistant",
    page_icon="🎓",
    layout="wide",
)


def render_study_plan(plan) -> None:
    st.subheader("Your study plan")
    st.write(plan.summary)

    for day in plan.days:
        st.markdown(f"**{day.date}**")
        if day.sessions:
            st.dataframe(
                [
                    {
                        "Subject": session.subject,
                        "Duration": f"{session.duration_minutes} min",
                        "Topic or goal": session.topic_or_goal,
                        "Notes": session.notes or "",
                    }
                    for session in day.sessions
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("No study sessions scheduled.")

    if plan.recommendations:
        st.markdown("**Recommendations**")
        for recommendation in plan.recommendations:
            st.markdown(f"- {recommendation}")


def main() -> None:
    st.title("🎓 AI Academic Assistant")

    initialize_session()

    mode = st.radio("Mode", ["Study Planner", "Ask"], horizontal=True)

    if mode == "Ask":
        from src.ui.rag_ui import load_rag_retriever
        from src.ui.workflow import load_workflow

        retriever = load_rag_retriever()
        workflow = load_workflow(retriever)
        question = st.chat_input("Ask your academic question...")

        if question:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):
                st.write(question)

            try:
                result = workflow.invoke(
                    {
                        "question": question,
                        "memory": st.session_state.memory,
                    }
                )

                answer = result.get("answer")

                with st.chat_message("assistant"):
                    if answer:
                        st.write(answer)
                    else:
                        st.write("I could not generate an answer.")

            except ValueError as exc:
                st.error(str(exc))

            except RuntimeError:
                st.error(
                    "The assistant could not generate a response. "
                    "Please try again."
                )

            except KeyError:
                st.error(
                    "The request is missing required information."
                )

            except Exception:
                st.error(
                    "Something went wrong while processing "
                    "your request."
                )

    else:
        st.header("Study Planner")
        subjects_text = st.text_input(
            "Subjects",
            placeholder="Mathematics, DBMS, Physics",
        )
        subjects = [
            subject.strip()
            for subject in subjects_text.split(",")
            if subject.strip()
        ]
        start_date = st.date_input(
            "Plan start date",
            value=date.today(),
        )

        if len(set(subjects)) != len(subjects):
            st.error("Enter each subject only once.")
        elif subjects:
            with st.form("study_plan_form"):
                st.markdown("**Exam dates**")
                default_exam_date = max(
                    start_date,
                    date.today() + timedelta(days=14),
                )
                exam_dates = {
                    subject: st.date_input(
                        subject,
                        value=default_exam_date,
                        key=f"exam_date_{index}",
                    )
                    for index, subject in enumerate(subjects)
                }
                available_hours = st.number_input(
                    "Available study hours per day",
                    min_value=0.5,
                    max_value=16.0,
                    value=3.0,
                    step=0.5,
                )
                weak_subjects = st.multiselect(
                    "Subjects to prioritize",
                    options=subjects,
                )
                preferred_study_times = st.multiselect(
                    "Preferred study times",
                    options=["Morning", "Afternoon", "Evening", "Night"],
                )
                additional_instructions = st.text_area(
                    "Additional instructions",
                    placeholder="For example: leave Sundays free",
                )
                generate_submitted = st.form_submit_button(
                    "Generate study plan"
                )

            if generate_submitted:
                request = StudyPlanRequest(
                    subjects=subjects,
                    exam_dates={
                        subject: exam_date.isoformat()
                        for subject, exam_date in exam_dates.items()
                    },
                    available_hours_per_day=available_hours,
                    start_date=start_date.isoformat(),
                    weak_subjects=weak_subjects,
                    preferred_study_times=preferred_study_times,
                    additional_instructions=(
                        additional_instructions.strip() or None
                    ),
                )
                try:
                    st.session_state.current_plan = generate_study_plan(
                        request
                    )
                    st.session_state.study_plan_request = request
                except Exception as exc:
                    st.error(f"Could not generate the study plan: {exc}")
        else:
            st.info("Enter your subjects to set up a study plan.")

        current_plan = st.session_state.current_plan
        saved_request = getattr(
            st.session_state,
            "study_plan_request",
            None,
        )

        if current_plan is not None:
            render_study_plan(current_plan)

        if current_plan is not None and saved_request is not None:
            with st.form("modify_study_plan_form"):
                modification = st.text_input(
                    "How should the plan change?",
                    placeholder="Give DBMS more time this week",
                )
                modify_submitted = st.form_submit_button("Update plan")

            if modify_submitted:
                try:
                    st.session_state.current_plan = modify_study_plan(
                        current_plan=current_plan,
                        modification_request=modification,
                        request=saved_request,
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not update the study plan: {exc}")


if __name__ == "__main__":
    main()