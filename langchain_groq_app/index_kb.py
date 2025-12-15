import os
from pathlib import Path
from typing import List

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS


def load_knowledge_texts(kb_dir: Path) -> List[str]:
    texts = []
    for p in sorted(kb_dir.glob("**/*")):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            with open(p, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts


def build_and_save_faiss(kb_dir: str = "knowledge_base", output_dir: str = "kb_faiss"):
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_path}")

    texts = load_knowledge_texts(kb_path)
    if not texts:
        raise ValueError("No text files found in knowledge_base/ to index.")

    print(f"Indexando {len(texts)} documento(s) desde {kb_path}")

    # Use sentence-transformers model as requested
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    faiss_store = FAISS.from_texts(texts, embeddings)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss_store.save_local(str(out))
    print(f"Índice FAISS guardado en: {out.resolve()}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Indexar knowledge_base/ a FAISS usando all-MiniLM-L6-v2")
    p.add_argument("--kb_dir", default="knowledge_base", help="Carpeta con archivos de la base de conocimientos")
    p.add_argument("--out", default="kb_faiss", help="Directorio destino para guardar el índice FAISS")
    args = p.parse_args()
    build_and_save_faiss(args.kb_dir, args.out)
