# AI Academic Assistant

An AI-powered academic assistant for college students. The application combines
retrieval-augmented generation (RAG) with an interactive study planner so that
students can ask questions about college documents and create or update
personalized study schedules from one Streamlit interface.

## What the project does

The assistant supports two modes:

### 1. Academic Q&A

- Loads college PDF documents from `data/sample_documents/`.
- Creates normalized text embeddings with
  `sentence-transformers/all-MiniLM-L6-v2`.
- Stores the embeddings and document metadata in a local FAISS index.
- Retrieves the most relevant document content for a question.
- Uses an OpenRouter-hosted chat model to generate an answer grounded in the
  retrieved context.
- Maintains the conversation in the Streamlit session.

### 2. Study planner

- Accepts subjects, exam dates, available study hours, weak subjects, preferred
  study times, and additional instructions.
- Generates a day-by-day study plan using the configured language model.
- Displays sessions with duration, topic or goal, and notes.
- Allows an existing plan to be modified using a natural-language request, such
  as “give DBMS more time this week”.

The sample knowledge base includes academic regulations, curriculum-related
information, internship guidance, and college reference material. Add or replace
PDFs in `data/sample_documents/` to customize the knowledge base.

## Technology stack

| Area | Technology |
| --- | --- |
| User interface | Python, Streamlit |
| LLM orchestration | LangChain, LangGraph |
| Language model provider | OpenRouter (`langchain-openrouter`) |
| Text embeddings | Sentence Transformers, `all-MiniLM-L6-v2` |
| Vector search | FAISS (`faiss-cpu`) |
| PDF ingestion | `pypdf`, LangChain community loaders |
| Configuration | `python-dotenv` |
| Testing and evaluation | Pytest-compatible tests and retrieval evaluation scripts |

## Project structure

```text
College_ChatBot/
├── app.py                         # Streamlit entry point
├── data/
│   ├── sample_documents/          # Source PDF documents for the RAG index
│   └── faiss_index/               # Generated index; not committed to Git
├── scripts/
│   └── build_faiss_index.py       # Builds the local FAISS index
├── src/
│   ├── llm/                       # Model configuration, prompts, memory, planner
│   ├── rag/                       # PDF loading, embeddings, retrieval, FAISS store
│   ├── ui/                        # Streamlit session, RAG, and workflow integration
│   └── workflow/                  # LangGraph state, routing, and workflow nodes
├── tests/                         # Retrieval and LLM tests/evaluation utilities
├── docs/
│   └── interfaces.md              # Module interface notes
├── .env.example                   # Environment variable template
├── requirements.txt               # Python dependencies
└── README.md
```

## Prerequisites

- Python 3.10 or newer
- An OpenRouter API key
- Internet access the first time the embedding model is downloaded

## Setup

1. Clone the repository and move into the project directory:

   ```bash
   git clone <repository-url>
   cd College_ChatBot
   ```

2. Create and activate a virtual environment:

   **Windows PowerShell**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create a local `.env` file from `.env.example` and set your API key:

   ```text
   OPENROUTER_API_KEY=your_openrouter_api_key
   LLM_PROVIDER=openrouter
   LLM_MODEL=openrouter/free
   LLM_TEMPERATURE=0.1
   LLM_MAX_TOKENS=700
   ```

   `OPENROUTER_API_KEY` is required for academic answers and study-plan
   generation. Do not commit `.env` or any API keys.

## Build the document index

The Q&A mode expects a generated FAISS index at `data/faiss_index/`. Build it
after adding or changing source PDFs:

```bash
python scripts/build_faiss_index.py
```

This creates:

- `data/faiss_index/index.faiss` — the searchable vector index
- `data/faiss_index/documents.json` — document text and source metadata

Generated indexes are ignored by Git and should be rebuilt when the source
documents change.

## Run the application

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.
Choose **Ask** for document-grounded academic questions or **Study Planner** to
create and update a schedule.

## Testing and evaluation

Run the automated tests with:

```bash
pytest
```

The retrieval evaluation script reports source hit rates at different values of
`k`:

```bash
python tests/retrieval_evaluation.py
```

The first run may download the Sentence Transformers model. Tests that exercise
the live language model also require a valid `OPENROUTER_API_KEY`.

## Configuration notes

The application currently supports the `openrouter` provider. Model behavior is
controlled through the following environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Authentication for OpenRouter | Required |
| `LLM_PROVIDER` | LLM provider name | `openrouter` |
| `LLM_MODEL` | OpenRouter model identifier | `openrouter/free` |
| `LLM_TEMPERATURE` | Response randomness | `0.1` |
| `LLM_MAX_TOKENS` | Maximum generated tokens | `700` |

## Important considerations

- Answers are based on the documents available in the local FAISS index. Keep
  the source PDFs current and rebuild the index after updates.
- The assistant should be used as an academic aid, not as a replacement for
  official notices, faculty guidance, or college administration.
- The first-time embedding-model download and local index generation can take
  several minutes depending on the machine and network.
- API keys and generated vector indexes are local runtime artifacts and should
  not be committed.

## Development workflow

The code is organized into focused modules for document processing, retrieval,
LLM integration, workflow routing, and the Streamlit UI. Team members can work
on separate branches and merge changes into `main` through pull requests.

When changing the document set or retrieval behavior, rebuild the FAISS index
and run both the automated tests and retrieval evaluation before opening a pull
request.
