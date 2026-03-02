from pathlib import Path

from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

DATASET_NAME = "akhooli/quran-simple-text"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = Path(__file__).resolve().parent / "quran_vectorstore"


def to_document(row: dict) -> Document:
    if {"number", "aya", "name", "text"}.issubset(row):
        surah = int(row["number"])
        ayah = int(row["aya"])
        surah_name = str(row["name"])
        arabic_text = str(row["text"])
        diacritized_text = str(row.get("d_text", ""))
        english_text = str(row.get("english", ""))
    elif {"surah", "ayah", "surah_name", "arabic"}.issubset(row):
        # Fallback if a different dataset variant is used.
        surah = int(row["surah"])
        ayah = int(row["ayah"])
        surah_name = str(row["surah_name"])
        arabic_text = str(row["arabic"])
        diacritized_text = ""
        english_text = str(row.get("english", ""))
    else:
        keys = ", ".join(sorted(row.keys()))
        raise KeyError(f"Unsupported dataset schema. Available keys: {keys}")

    lines = [
        f"Surah {surah}:{ayah} ({surah_name})",
        "",
        "Arabic:",
        arabic_text,
    ]
    if diacritized_text and diacritized_text != arabic_text:
        lines.extend(["", "Arabic (diacritized):", diacritized_text])
    if english_text:
        lines.extend(["", "English:", english_text])

    metadata = {
        "surah": surah,
        "ayah": ayah,
        "surah_name": surah_name,
    }
    if "class" in row:
        metadata["class"] = str(row["class"])
    if "rev_order" in row:
        metadata["rev_order"] = int(row["rev_order"])

    return Document(page_content="\n".join(lines).strip(), metadata=metadata)


def main() -> None:
    ds = load_dataset(DATASET_NAME, split="train")
    if len(ds) == 0:
        raise RuntimeError(f"No rows loaded from dataset: {DATASET_NAME}")

    documents = [to_document(row) for row in ds]
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = FAISS.from_documents(documents, embeddings)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print(f"Vector store created and saved to: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
