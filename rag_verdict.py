import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = Path(__file__).resolve().parent / "quran_vectorstore"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = """
You are an evidence-based Islamic fact-checking assistant.

Use ONLY the provided context verses.
Do NOT invent verses.
If insufficient evidence, say "Not supported by provided evidence".
If interpretation is required, say "Interpretation required".

Return answer as valid JSON with this exact schema:
{{
  "verdict": "",
  "quoted_verses": [],
  "explanation": "",
  "confidence": ""
}}

Context:
{context}
""".strip()


def build_chain():
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTORSTORE_DIR}. Run ingestion.py first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        # Safe for local/trusted indices only.
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Claim:\n{input}"),
        ]
    )

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)


if __name__ == "__main__":
    try:
        chain = build_chain()
        query = "Does the Qur'an allow forcing someone to convert?"
        result = chain.invoke({"input": query})
        print(result["answer"])
    except Exception as exc:
        message = str(exc)
        if "Failed to establish a new connection" in message or "localhost', port=11434" in message:
            raise SystemExit(
                "Error: Could not connect to Ollama at http://localhost:11434. "
                "Start Ollama, then run `ollama pull llama3.2` and retry."
            )
        if "model" in message and "not found" in message.lower():
            raise SystemExit(
                f"Error: Ollama model '{OLLAMA_MODEL}' is not available locally. "
                f"Run `ollama pull {OLLAMA_MODEL}` and retry."
            )
        raise SystemExit(f"Error: {message}")
