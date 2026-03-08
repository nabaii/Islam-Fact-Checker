import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv(Path(__file__).resolve().parent / "global.env")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = Path(__file__).resolve().parent / "quran_vectorstore"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """
You are an evidence-based Islamic fact-checking assistant.

Use ONLY the provided context verses.
Do NOT invent verses.
If insufficient evidence, say "Not supported by provided evidence".
If interpretation is required, say "Interpretation required".

For every quoted verse, include BOTH the Arabic text and an English translation.
If the context only has Arabic, provide a widely accepted English translation yourself.

Return answer as valid JSON with this exact schema:
{{
  "verdict": "",
  "quoted_verses": [
    {{
      "reference": "Surah X:Y",
      "arabic": "",
      "english": ""
    }}
  ],
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
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Set it in your current terminal and retry."
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

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)


if __name__ == "__main__":
    try:
        chain = build_chain()
        query = "Does the Qur'an allow forcing someone to convert?"
        result = chain.invoke({"input": query})
        answer = result["answer"]
        try:
            parsed = json.loads(answer)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(answer)
    except Exception as exc:
        message = str(exc)
        if "OPENAI_API_KEY" in message:
            raise SystemExit(
                "Error: OPENAI_API_KEY is not set in this terminal session."
            )
        if "insufficient_quota" in message or "You exceeded your current quota" in message:
            raise SystemExit(
                "Error: OpenAI quota exceeded. Check billing/usage and retry."
            )
        raise SystemExit(f"Error: {message}")
