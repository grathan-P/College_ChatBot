
from langgraph.graph import END, START, StateGraph

from src.rag.retriever import Retriever
from src.workflow.nodes import (
    create_study_plan,
    generate_academic_answer,
    modify_existing_study_plan,
    retrieve_documents,
)
from src.workflow.router import route_request
from src.workflow.state import WorkflowState


def route_workflow(state: WorkflowState) -> str:
    """
    Determine which workflow branch should handle the request.
    """
    return route_request(state["question"])


def route_after_intent(state: WorkflowState) -> str:
    """
    Return the LangGraph branch corresponding to the detected intent.
    """
    intent = state["intent"]

    if intent == "academic":
        return "academic"

    if intent == "study_plan_create":
        return "study_plan_create"

    if intent == "study_plan_modify":
        return "study_plan_modify"

    raise ValueError(f"Unsupported workflow intent: {intent}")


def build_workflow(retriever: Retriever):
    """
    Build and compile the College Academic Assistant workflow.

    Args:
        retriever: Person 1's RAG Retriever instance.

    Returns:
        A compiled LangGraph workflow.
    """
    graph = StateGraph(WorkflowState)

    def detect_intent(state: WorkflowState) -> dict:
        intent = route_workflow(state)

        return {
            "intent": intent,
        }

    def retrieve_node(state: WorkflowState) -> dict:
        return retrieve_documents(
            state,
            retriever,
        )

    graph.add_node(
        "detect_intent",
        detect_intent,
    )

    graph.add_node(
        "retrieve_documents",
        retrieve_node,
    )

    graph.add_node(
        "generate_academic_answer",
        generate_academic_answer,
    )

    graph.add_node(
        "create_study_plan",
        create_study_plan,
    )

    graph.add_node(
        "modify_study_plan",
        modify_existing_study_plan,
    )

    graph.add_edge(
        START,
        "detect_intent",
    )

    graph.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "academic": "retrieve_documents",
            "study_plan_create": "create_study_plan",
            "study_plan_modify": "modify_study_plan",
        },
    )

    graph.add_edge(
        "retrieve_documents",
        "generate_academic_answer",
    )

    graph.add_edge(
        "generate_academic_answer",
        END,
    )

    graph.add_edge(
        "create_study_plan",
        END,
    )

    graph.add_edge(
        "modify_study_plan",
        END,
    )

    return graph.compile()
