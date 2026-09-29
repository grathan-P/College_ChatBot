# Module Interfaces

## Person 1: RAG
File: src/rag/retriever.py

Function:
retrieve_context(question: str)

Returns:
A list of relevant document chunks with their
source metadata.

## Person 2: LLM
File: src/llm/gemini_client.py

Function:
generate_answer(question, context, history)

Returns:
A text response grounded in the retrieved context.

## Person 3: LangGraph
File: src/workflow/graph.py

Function:
run_workflow(question, history)

Returns:
A structured result containing the response
and any relevant study-plan updates.

## Person 4: Streamlit
File: app.py

Responsibilities:
- Collect user input
- Display assistant responses
- Maintain chat history
- Connect the UI to the workflow