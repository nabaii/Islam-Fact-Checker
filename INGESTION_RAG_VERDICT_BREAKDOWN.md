# Detailed Breakdown of `ingestion.py` and `rag_verdict.py`

## 1) High-Level System Overview

This project has a two-stage Retrieval-Augmented Generation (RAG) pipeline for Qur'an-based claim checking:

1. `ingestion.py` builds a local FAISS vector index from a Qur'an dataset.
2. `rag_verdict.py` loads that index, retrieves relevant verses for a claim, and asks an OpenAI LLM to produce a structured verdict.

In short:

- `ingestion.py`: data preparation + indexing
- `rag_verdict.py`: retrieval + LLM reasoning + structured output

---

## 2) `ingestion.py` Breakdown

### Purpose

`ingestion.py` converts dataset rows into LangChain `Document` objects, embeds them with a multilingual sentence-transformer model, and stores them in a FAISS index on disk.

### Imports and Dependency Strategy

- `Path` from `pathlib`: builds filesystem-safe paths.
- `load_dataset` from `datasets`: downloads/loads Hugging Face dataset rows.
- `FAISS` from `langchain_community.vectorstores`: vector index backend.
- `Document` from `langchain_core.documents`: standard document wrapper for content + metadata.
- `HuggingFaceEmbeddings`: imported from `langchain_huggingface` if installed; otherwise falls back to `langchain_community.embeddings`.

The fallback import pattern keeps compatibility across LangChain package split versions.

### Constants

- `DATASET_NAME = "akhooli/quran-simple-text"`
  - Hugging Face dataset identifier.
- `EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`
  - Embedding model used for semantic search.
- `VECTORSTORE_DIR = Path(__file__).resolve().parent / "quran_vectorstore"`
  - Output folder where FAISS index files are saved.

### `to_document(row: dict) -> Document`

This is the core row-normalization function.

#### Schema Handling

It supports two row formats:

1. Primary schema (expected):
   - required keys: `number`, `aya`, `name`, `text`
   - optional: `d_text`, `english`, `class`, `rev_order`
2. Fallback schema (alternate variant):
   - required keys: `surah`, `ayah`, `surah_name`, `arabic`
   - optional: `english`

If neither schema matches, it raises:

- `KeyError("Unsupported dataset schema. Available keys: ...")`

This makes schema issues fail fast and visible.

#### Field Normalization

All key fields are cast to stable types:

- surah/ayah -> `int`
- names/text -> `str`

This prevents inconsistent typing from upstream dataset rows.

#### Page Content Construction

It builds a readable multiline `page_content`:

1. Header line: `Surah {surah}:{ayah} ({surah_name})`
2. Arabic text section
3. Optional diacritized Arabic section (only if present and different from base Arabic)
4. Optional English section

This structure improves retrieval quality by keeping all relevant verse text variants in one retrievable chunk.

#### Metadata Construction

Always includes:

- `surah`
- `ayah`
- `surah_name`

Conditionally includes:

- `class` (if present)
- `rev_order` (if present)

This metadata can be used later for filtering, ranking, audit trails, or UI display.

### `main()`

Execution flow:

1. `load_dataset(DATASET_NAME, split="train")`
2. Validate non-empty dataset; otherwise `RuntimeError`.
3. Convert each row via `to_document`.
4. Initialize embedding model (`HuggingFaceEmbeddings`).
5. Build FAISS index: `FAISS.from_documents(documents, embeddings)`.
6. Ensure output directory exists (`mkdir(parents=True, exist_ok=True)`).
7. Persist index with `save_local`.
8. Print saved path.

### Output Artifact

Running `ingestion.py` creates `quran_vectorstore/` next to the script, containing FAISS index data (typically `.faiss` + metadata pickle files).

### Failure Modes in `ingestion.py`

- Dataset unavailable/network/auth issues -> `load_dataset` fails.
- Dataset schema changed -> `KeyError` from `to_document`.
- Empty dataset split -> `RuntimeError`.
- Embedding model download/init issues -> embedding initialization fails.
- Write permission/path issues -> save step fails.

---

## 3) `rag_verdict.py` Breakdown

### Purpose

`rag_verdict.py` loads the saved vector store, retrieves top-k relevant verse documents for a claim, sends context + claim to an OpenAI chat model, and returns a JSON-formatted verdict.

### Imports and Dependency Strategy

- `os`, `Path`: environment variables + filesystem paths.
- `FAISS`: load stored vector index.
- `ChatPromptTemplate`: structured prompt building.
- `HuggingFaceEmbeddings` with same fallback strategy as ingestion.
- `ChatOpenAI` from `langchain_openai`.
- `create_retrieval_chain` and `create_stuff_documents_chain` from `langchain_classic`.

Using the same embedding model as ingestion is critical; mismatch can degrade retrieval.

### Constants and Runtime Config

- `EMBEDDING_MODEL`: same multilingual model as ingestion.
- `VECTORSTORE_DIR`: same directory path (`quran_vectorstore`).
- `OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")`
  - Configurable OpenAI model name.
- `OPENAI_API_KEY`
  - Required environment variable for authentication.

### `SYSTEM_PROMPT`

The prompt enforces strict behavior:

- Use only retrieved verses.
- Do not invent verses.
- Emit fallback language when evidence is weak or interpretation is needed.
- Return strict JSON schema:
  - `verdict`
  - `quoted_verses`
  - `explanation`
  - `confidence`

It also injects retrieved context with `{context}` placeholder.

### `build_chain()`

This function composes the full RAG chain.

#### Step-by-step

1. Guard: if `VECTORSTORE_DIR` missing -> `FileNotFoundError` with guidance to run ingestion.
2. Guard: if `OPENAI_API_KEY` missing -> explicit environment error.
3. Create embeddings object using same model.
4. Load FAISS index:
   - `allow_dangerous_deserialization=True`
   - Intended only for trusted/local indices.
5. Create retriever: `as_retriever(search_kwargs={"k": 5})`
   - Retrieves top 5 semantically nearest documents.
6. Build chat prompt messages:
   - system message = policy + format
   - human message = `Claim:\n{input}`
7. Initialize OpenAI chat model:
   - `model=OPENAI_MODEL`
   - `temperature=0` for deterministic responses.
8. Build QA combine chain with `create_stuff_documents_chain`
   - "Stuff" means all retrieved docs are concatenated into prompt context.
9. Return retrieval chain from `create_retrieval_chain(retriever, qa_chain)`.

### `__main__` Execution Block

Default behavior when run directly:

1. Builds chain.
2. Uses sample claim:
   - `"Does the Qur'an allow forcing someone to convert?"`
3. Invokes chain with `{"input": query}`.
4. Prints `result["answer"]` (model output, expected JSON string).

### Error Handling in `__main__`

The script traps common failures and gives actionable messages:

- `OPENAI_API_KEY` missing:
  - advises setting the key in the active terminal session.
- OpenAI quota exceeded:
  - advises checking account billing/usage.
- Any other exception:
  - re-raised as `SystemExit("Error: ...")`.

This produces clean CLI errors instead of noisy tracebacks.

---

## 4) End-to-End Data/Control Flow

1. Run `ingestion.py`.
2. Dataset rows become standardized `Document` objects.
3. Documents become vectors in FAISS index.
4. Run `rag_verdict.py`.
5. Claim query is embedded.
6. Retriever fetches top 5 nearest verse documents.
7. Retrieved text is inserted into strict system prompt context.
8. OpenAI model generates JSON verdict based only on provided context.

---

## 5) Important Coupling Between the Two Files

These must stay aligned:

- Same embedding model in both scripts (`EMBEDDING_MODEL`).
- Same vectorstore directory (`VECTORSTORE_DIR`).
- Ingestion must run before retrieval.

If either model/path diverges, retrieval quality or loading behavior breaks.

---

## 6) Current Design Strengths

- Clear separation between indexing and inference.
- Schema-aware ingestion with explicit validation.
- Compatibility fallbacks for LangChain package changes.
- Deterministic LLM setting (`temperature=0`) for stable outputs.
- Explicit user-facing error messages for common OpenAI setup issues.

---

## 7) Limitations and Risks

- `allow_dangerous_deserialization=True` should only be used with trusted local files.
- JSON format is instructed but not programmatically validated after model output.
- No citation post-processing layer to ensure `quoted_verses` exactly match retrieved docs.
- `k=5` is fixed and may under/over-provide context depending on claim complexity.
- No reranking step; retrieval quality fully depends on embedding nearest-neighbor search.

---

## 8) Practical Run Sequence

From project root:

```powershell
python ingestion.py
python rag_verdict.py
```

Optional environment variables:

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_MODEL = "gpt-4o-mini"
python rag_verdict.py
```

---

## 9) Summary

- `ingestion.py` builds the searchable Qur'an knowledge base.
- `rag_verdict.py` retrieves relevant verses and asks an OpenAI model to produce a structured verdict.
- Together they implement an evidence-constrained RAG verifier with clear operational boundaries.
