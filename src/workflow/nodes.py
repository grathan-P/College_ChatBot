
from src.rag.retriever import Retriever
from src.llm.generator import generate_answer
from src.llm.planner import (
    generate_study_plan,
    modify_study_plan,
)


def retrieve_documents(state: dict, retriever: Retriever) -> dict:
    """
    Retrieve relevant documents for the user's academic question.
    """
    question = state["question"]

    results = retriever.retrieve(
        query=question,
        top_k=5,
    )

    return {
        "retrieved_documents": results,
    }


def generate_academic_answer(state: dict) -> dict:
    """
    Generate a grounded academic answer using Person 2's LLM interface.
    """
    generated = generate_answer(
        question=state["question"],
        retrieval_results=state["retrieved_documents"],
        memory=state.get("memory"),
    )

    return {
        "answer": generated.answer,
        "sources": generated.sources,
    }


def create_study_plan(state: dict) -> dict:
    """
    Generate a personalized study plan using Person 2's planner.
    """
    request = state["study_plan_request"]

    plan = generate_study_plan(
        request=request,
    )

    return {
        "study_plan": plan,
    }


def modify_existing_study_plan(state: dict) -> dict:
    """
    Modify an existing study plan using Person 2's planner.
    """
    current_plan = state["current_plan"]
    request = state["study_plan_request"]
    modification_request = state["question"]

    updated_plan = modify_study_plan(
        current_plan=current_plan,
        modification_request=modification_request,
        request=request,
    )

    return {
        "study_plan": updated_plan,
    }
